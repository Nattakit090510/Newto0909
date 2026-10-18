"""
Northbridge Bank - Credit Risk Query Engine (Streamlit)

Natural-language query engine for the commercial lending portfolio:
  1. Intent classification -> verified template (VQ1-VQ10) or generated SQL
  2. Read-only, validated SQL (5-check validation gate, one retry on the generated track)
  3. Execution against a read-only SQLite connection
  4. Narrative response + SQL + raw data + confidence score
  5. Audit trail for every analytical output (audit_log.jsonl)

Run locally:   streamlit run app.py
"""

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np  # noqa: F401  (kept for parity with the notebook)
import pandas as pd
import sqlparse
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate  # noqa: F401  (kept for parity with the notebook)
from langchain_openai import ChatOpenAI

# --------------------------------------------------------------------------------------
# PAGE SETUP
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Northbridge Bank - Portfolio Analytics", page_icon="🏦", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = os.environ.get("DB_PATH", str(BASE_DIR / "credit_risk_portfolio.db"))
TEST_QUERIES_PATH = os.environ.get("TEST_QUERIES_PATH", str(BASE_DIR / "test_queries.csv"))
AUDIT_LOG_PATH = os.environ.get("AUDIT_LOG_PATH", str(BASE_DIR / "audit_log.jsonl"))
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
EVALUATOR_MODEL = os.environ.get("EVALUATOR_MODEL", "gpt-4o")


# --------------------------------------------------------------------------------------
# CREDENTIALS  (st.secrets -> environment variables -> local config.json)
# --------------------------------------------------------------------------------------
def load_credentials():
    """Load the OpenAI-compatible API key/base URL and expose them as environment variables
    (ChatOpenAI reads OPENAI_API_KEY and OPENAI_BASE_URL). Never hard-code keys in this file."""
    api_key, base_url = None, None

    # 1) Streamlit secrets (.streamlit/secrets.toml locally, App settings -> Secrets on Streamlit Cloud)
    try:
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
            base_url = st.secrets.get("OPENAI_API_BASE", None)
    except Exception:
        pass

    # 2) Environment variables
    if not api_key and os.environ.get("OPENAI_API_KEY"):
        api_key = os.environ["OPENAI_API_KEY"]
        base_url = (
            base_url
            or os.environ.get("OPENAI_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
        )

    # 3) Local config.json (same format the notebook used)
    if not api_key:
        config_path = BASE_DIR / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                api_key = config.get("OPENAI_API_KEY")
                base_url = config.get("OPENAI_API_BASE")
            except Exception:
                pass

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    return bool(api_key)


@st.cache_resource(show_spinner=False)
def get_llms():
    """Initialise the generation LLM and the (stronger) evaluator LLM once per server process."""
    llm_ = ChatOpenAI(model=LLM_MODEL, temperature=0)
    evaluator_llm_ = ChatOpenAI(model=EVALUATOR_MODEL, temperature=0)
    return llm_, evaluator_llm_


CREDENTIALS_OK = load_credentials()
llm, evaluator_llm = get_llms() if CREDENTIALS_OK else (None, None)


# --------------------------------------------------------------------------------------
# DATABASE  (read-only connection, one per browser session)
# --------------------------------------------------------------------------------------
def get_connection():
    if "db_conn" not in st.session_state:
        uri = f"file:{quote(Path(DB_PATH).resolve().as_posix(), safe='/:')}?mode=ro"
        st.session_state["db_conn"] = sqlite3.connect(uri, uri=True, check_same_thread=False)
    return st.session_state["db_conn"]


# --------------------------------------------------------------------------------------
# DATABASE SCHEMA (provided to the LLM in the prompts)
# --------------------------------------------------------------------------------------
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""


# --------------------------------------------------------------------------------------
# VERIFIED QUERY TEMPLATE LIBRARY
# --------------------------------------------------------------------------------------
# VQ1: Sector-wise Outstanding and NPA Breakdown
sql_1 = """
SELECT
    s.sector_name,
    SUM(l.total_outstanding) / 1000000.0 as total_outstanding_million,
    SUM(CASE WHEN l.asset_classification IN ('Substandard', 'Doubtful', 'Loss') THEN l.total_outstanding ELSE 0 END) / 1000000.0 as npa_outstanding_million
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
GROUP BY s.sector_name
ORDER BY total_outstanding_million DESC;
"""

# VQ2: Portfolio Outstanding by Loan Category
sql_2 = """
SELECT
    loan_category,
    SUM(total_outstanding) / 1000000.0 as total_outstanding_million,
    COUNT(loan_account_number) as loan_count
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_million DESC;
"""

# VQ3: IFRS 9 Stage-wise ECL Summary
sql_3 = """
SELECT
    ifrs9_stage,
    COUNT(loan_account_number) as loan_count,
    SUM(ead_amount) / 1000000.0 as total_ead_million,
    SUM(ecl_amount) / 1000000.0 as total_ecl_million
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage;
"""

# VQ4: Provision Coverage Ratio by Sector
sql_4 = """
SELECT
    s.sector_name,
    AVG(p.provision_coverage_ratio) as avg_provision_coverage_ratio
FROM provisioning p
JOIN loan_master l ON p.loan_account_number = l.loan_account_number
JOIN sector_master s ON l.sector_code = s.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY s.sector_name
ORDER BY avg_provision_coverage_ratio DESC;
"""

# VQ5: Top 10 Loan Exposures
sql_5 = """
SELECT
    l.borrower_name,
    s.sector_name,
    l.total_outstanding / 1000000.0 as total_outstanding_million,
    l.asset_classification
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
ORDER BY total_outstanding_million DESC
LIMIT 10;
"""

# VQ6: Top 5 Business Group Exposures
sql_6 = """
SELECT
    group_name,
    COUNT(loan_account_number) as loan_count,
    SUM(total_outstanding) / 1000000.0 as total_outstanding_million
FROM loan_master
WHERE group_name IS NOT NULL
GROUP BY group_name
ORDER BY total_outstanding_million DESC
LIMIT 5;
"""

# VQ7: All Overdue Loan Accounts
sql_7 = """
SELECT
    l.loan_account_number,
    l.borrower_name,
    s.sector_name,
    l.total_outstanding / 1000000.0 as total_outstanding_million,
    l.days_past_due,
    l.asset_classification
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
WHERE l.days_past_due > 0
ORDER BY l.days_past_due DESC;
"""

# VQ8: DPD Bucket Distribution
sql_8 = """
SELECT
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END as dpd_bucket,
    COUNT(loan_account_number) as loan_count,
    SUM(total_outstanding) / 1000000.0 as total_outstanding_million
FROM loan_master
GROUP BY dpd_bucket
ORDER BY MIN(days_past_due);
"""

# VQ9: Latest Rating Downgrades
sql_9 = """
SELECT
    borrower_id,
    previous_rating,
    internal_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
  AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC;
"""

# VQ10: ECL Trend Across Reporting Quarters
sql_10 = """
SELECT
    reporting_date,
    SUM(ecl_amount) / 1000000.0 as total_ecl_million
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date;
"""

verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1
    },

    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2
    },

    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3
    },

    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4
    },

    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': sql_5
    },

    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6
    },

    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7
    },

    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8
    },

    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9
    },

    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': sql_10
    }
}


# --------------------------------------------------------------------------------------
# TOOL 1: INTENT CLASSIFICATION
# --------------------------------------------------------------------------------------
def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.

    Returns:
    - dict: Contains 'route' (verified or generated),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
Analyze the user's question and match it to the Verified Query Library if it closely aligns with an existing template's purpose.
If there is a match, set route to 'verified' and specify the query_id.
If it requires custom logic, set route to 'generated' and query_id to null.

User Question:
{user_question}

Verified Query Library:
{library_descriptions}

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


# --------------------------------------------------------------------------------------
# TOOL 2: QUERY GENERATION
# --------------------------------------------------------------------------------------
def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.
    '''
    generation_prompt = f"""
    You are an expert SQL developer. Write a valid SQLite read-only query to answer the user's question.
    Use ONLY the tables and columns provided in the schema context below.

    User Question:
    {user_question}

    Database Schema:
    {schema_context}

    ### CRITICAL INSTRUCTION
    Return ONLY the raw SQL code.
    Do NOT wrap the code in markdown blocks like ```sql.
    Do NOT write any explanations, greetings, introduction, or conversational filler.
    The query must begin directly with 'SELECT' or 'WITH'.
    """

    sql = llm.invoke(generation_prompt).content.strip()
    # Fallback cleanup patterns
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


# --------------------------------------------------------------------------------------
# TOOL 3: QUERY VALIDATION (five checks)
# --------------------------------------------------------------------------------------
def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
    Validates a candidate SQL query through five checks before execution.
    '''
    result = {'passed': False, 'failed_check': None, 'details': '', 'relevance_confidence': None}

    # Check 1: Read-only shape check
    clean_sql = candidate_sql.strip().rstrip(';')
    sql_upper = clean_sql.upper()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'], result['details'] = 'read_only_shape', 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'], result['details'] = 'read_only_shape', f'Forbidden keyword detected: {kw}'
            return result
    if ';' in clean_sql:
        result['failed_check'], result['details'] = 'read_only_shape', 'Multiple statements are not allowed'
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select','from','where','and','or','group','by','order','having','limit','join','on','as','case',
                    'when','then','else','end','sum','count','avg','min','max','round','desc','asc','left','right',
                    'inner','outer','distinct','null','is','not','in','like','with','union','all','between','coalesce'}

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {clean_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'], result['details'] = 'parse_plan_dry_run', f'SQL failed to parse or plan: {str(e)}'
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It may return broader data than requested. "
        "Judge only whether the underlying metric, tables, and logic match the question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be appropriately scoped."
    )

    relevance_prompt = f"""
You are evaluating a SQL query against a user question.
{track_context}

User Question: {user_question}
Candidate SQL: {candidate_sql}

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}
"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'], result['details'] = 'llm_relevance', f"Relevance failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check
    # NOTE: the SQL is wrapped in a sub-select before appending LIMIT 0. In the notebook, "<sql> LIMIT 0"
    # raised a syntax error for templates that already end in LIMIT (VQ5, VQ6) and wrongly escalated them.
    if is_verified_track:
        expected_sql = query_library[query_id]['sql'].strip().rstrip(';')
        try:
            expected_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({expected_sql}) LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({clean_sql}) LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'], result['details'] = 'template_integrity', f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'], result['details'] = 'template_integrity', f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


# --------------------------------------------------------------------------------------
# TOOL 4: RETRY GENERATION
# --------------------------------------------------------------------------------------
def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.
    '''
    retry_prompt = f"""
    You are an expert SQL developer. Your previously generated query failed validation.
    Please correct the SQL to satisfy the user's intent while fixing the error below.
    Ensure it is a valid SQLite read-only query.

    User Question:
    {user_question}

    Failed SQL:
    {failed_sql}

    Validation Error:
    {error_message}

    Database Schema:
    {schema_context}

    ### CRITICAL INSTRUCTION
    Return ONLY the raw corrected SQL code.
    Do NOT wrap the code in markdown blocks like ```sql.
    Do NOT write any explanations, introductions, or conversational filler.
    The query must begin directly with 'SELECT' or 'WITH'.
    """

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


# --------------------------------------------------------------------------------------
# TOOL 5: QUERY EXECUTION
# --------------------------------------------------------------------------------------
def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


# --------------------------------------------------------------------------------------
# TOOL 6: RESPONSE GENERATION
# --------------------------------------------------------------------------------------
def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.
    '''
    response_prompt = f"""
You are a bank's analytical assistant answering a business user.
Using the User Question and the returned Data Result, provide a concise, direct answer.
Highlight the key figures and metrics explicitly.

User Question:
{user_question}

Data Result:
{dataframe.to_string()}
"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


# --------------------------------------------------------------------------------------
# PIPELINE ORCHESTRATION
# --------------------------------------------------------------------------------------
def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - verbose (bool): If True, prints intermediate pipeline stages to the server console.

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
            (Streamlit additions: 'warnings' and 'trace' keys in the log.)
    '''

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None,
        'warnings': [],
        'trace': []
    }

    def say(message):
        log['trace'].append(message)
        if verbose:
            print(message)

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    say(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
    say(f"    Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    say(f"[2] Query Construction: {'loaded from library' if log['route']=='verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    say(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
    if not gate['passed']:
        say(f"    Failed check: {gate.get('failed_check')}")
        say(f"    Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        say(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        say(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            say(f"    Retry failed check: {gate.get('failed_check')}")
            say(f"    Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        say(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, **log}

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)
    log['warnings'] = exec_result['warnings']

    say(f"[4] Execute: {len(df)} rows returned")
    if exec_result['warnings']:
        say(f"    Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    say(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, **log}


# --------------------------------------------------------------------------------------
# AUDIT TRAIL (every analytical output is logged; result data itself is NOT stored)
# --------------------------------------------------------------------------------------
def write_audit_record(user_question, source, result=None, error=None):
    record = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'source': source,
        'user_question': user_question,
    }
    if result is not None:
        log = result['log']
        record.update({
            'status': 'escalated' if log['escalated'] else 'answered',
            'route': log['route'],
            'query_id': log['query_id'],
            'match_reason': log['match_reason'],
            'candidate_sql': log['candidate_sql'],
            'executed_sql': log['executed_sql'],
            'gate_result': log['gate_result'],
            'retry_used': log['retry_used'],
            'escalated': log['escalated'],
            'row_count': log['row_count'],
            'confidence': log['confidence'],
            'warnings': log['warnings'],
            'narrative': log['narrative'],
        })
    else:
        record.update({'status': 'error', 'error': error})

    st.session_state.setdefault('session_audit', []).append(record)
    try:
        with open(AUDIT_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, default=str) + '\n')
    except OSError:
        pass  # file system not writable; the in-session copy is still available


def load_audit_records():
    records = []
    if os.path.exists(AUDIT_LOG_PATH):
        try:
            with open(AUDIT_LOG_PATH, 'r', encoding='utf-8') as f:
                records = [json.loads(line) for line in f if line.strip()]
        except (OSError, ValueError):
            records = []
    return records or st.session_state.get('session_audit', [])


def execute_with_audit(user_question, db_connection, source='interactive'):
    """Run the pipeline, write the audit record, and return (result, error)."""
    try:
        result = run_pipeline(user_question, db_connection, verified_query_library, database_schema, verbose=False)
        write_audit_record(user_question, source, result=result)
        return result, None
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI and audit trail
        write_audit_record(user_question, source, error=str(exc))
        return None, exc


# --------------------------------------------------------------------------------------
# UI HELPERS
# --------------------------------------------------------------------------------------
def fmt_confidence(value):
    if isinstance(value, (int, float)) and not pd.isna(value):
        return f"{value:.2f}"
    return str(value) if value is not None else "N/A"


def safe_md(text):
    """Escape $ so dollar amounts are not rendered as LaTeX by Streamlit markdown."""
    return (text or "").replace("$", "\\$")


@st.cache_data(show_spinner=False)
def portfolio_summary(_conn):
    metrics_query = """
    SELECT
        COUNT(loan_account_number) as total_loans,
        SUM(total_outstanding) / 1000000.0 as total_outstanding_million,
        AVG(interest_rate) as average_interest_rate,
        SUM(CASE WHEN asset_classification IN ('Substandard', 'Doubtful', 'Loss') THEN total_outstanding ELSE 0 END) / 1000000.0 as npa_outstanding_million
    FROM loan_master;
    """
    summary = pd.read_sql_query(metrics_query, _conn)
    total_loans = int(summary['total_loans'].iloc[0])
    outstanding = float(summary['total_outstanding_million'].iloc[0])
    avg_rate = float(summary['average_interest_rate'].iloc[0])
    npa = float(summary['npa_outstanding_million'].iloc[0])
    npa_ratio = (npa / outstanding) * 100 if outstanding > 0 else 0
    return total_loans, outstanding, avg_rate, npa, npa_ratio


@st.cache_data(show_spinner=False)
def load_ground_truth(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def render_result(result, question):
    log = result['log']
    df = result['dataframe']

    st.markdown(f"**Question:** {safe_md(question)}")

    if log['escalated']:
        st.warning("This question could not be reliably resolved and has been **escalated to a human analyst**.")
        st.write(safe_md(log['narrative']))
    else:
        st.subheader("Answer")
        st.write(safe_md(log['narrative']))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Route", str(log['route']).capitalize())
    c2.metric("Verified Query ID", log['query_id'] or "-")
    c3.metric("Confidence", fmt_confidence(log['confidence']))
    c4.metric("Rows returned", "-" if log['row_count'] is None else f"{log['row_count']:,}")

    if log['retry_used']:
        st.info("The first generated query failed validation; the SQL was regenerated once (retry used).")
    for w in log.get('warnings', []):
        st.warning(w)

    with st.expander("SQL query used" if not log['escalated'] else "Candidate SQL (not executed)", expanded=True):
        st.code((log['executed_sql'] or log['candidate_sql'] or '').strip(), language="sql")

    if df is not None:
        with st.expander("Raw data returned", expanded=True):
            st.dataframe(df)
            st.download_button(
                "Download result as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="query_result.csv",
                mime="text/csv",
            )

    with st.expander("Routing & validation details"):
        st.markdown(f"**Match reason:** {safe_md(log['match_reason'])}")
        st.json(log['gate_result'])
        st.text("\n".join(log['trace']))


def run_evaluation(ground_truth, db_connection):
    evaluation_rows = []
    total = len(ground_truth)
    progress = st.progress(0.0, text="Running test cases...")

    for i, (_, gt) in enumerate(ground_truth.iterrows()):
        result, error = execute_with_audit(gt['User Query'], db_connection, source='evaluation')
        if result is None:
            tr = {'route': 'error', 'query_id': None, 'confidence': None, 'row_count': None, 'escalated': True}
        else:
            tr = result

        evaluation_rows.append({
            'Test Case': gt['Test Case'],
            'Expected Route': gt['Expected Route'],
            'Actual Route': tr['route'],
            'Route Match': tr['route'] == gt['Expected Route'],
            'Expected Query ID': gt['Expected Query ID'],
            'Actual Query ID': tr['query_id'],
            'Query ID Match': (
                pd.isna(gt['Expected Query ID']) and pd.isna(tr['query_id'])
            ) or tr['query_id'] == gt['Expected Query ID'],
            'Confidence': tr['confidence'],
            'Rows Returned': tr['row_count'],
            'Escalated': bool(tr['escalated']),
        })
        progress.progress((i + 1) / total, text=f"Completed {i + 1} of {total} test cases")

    progress.empty()
    evaluation_df = pd.DataFrame(evaluation_rows)

    path_accuracy = evaluation_df['Route Match'].mean() * 100

    verified = evaluation_df['Expected Route'].astype(str).str.strip().str.lower() == 'verified'
    query_accuracy = evaluation_df.loc[verified, 'Query ID Match'].mean() * 100 if verified.any() else float('nan')

    # Coerce string values like 'ESCALATED' to NaN so they are excluded from the mean
    evaluation_df['Confidence'] = pd.to_numeric(evaluation_df['Confidence'], errors='coerce')
    average_confidence = evaluation_df['Confidence'].mean()
    escalation_rate = evaluation_df['Escalated'].mean() * 100

    return {
        'df': evaluation_df,
        'path_accuracy': path_accuracy,
        'query_accuracy': query_accuracy,
        'average_confidence': average_confidence,
        'escalation_rate': escalation_rate,
    }


# --------------------------------------------------------------------------------------
# MAIN APP
# --------------------------------------------------------------------------------------
st.title("🏦 Northbridge Bank Credit Risk Query Engine")
st.write("Ask commercial lending portfolio and credit-risk questions in plain English. "
         "Every answer shows the SQL used, the raw data, and a confidence score, and is written to the audit trail.")

# --- Startup checks -----------------------------------------------------------------
if not CREDENTIALS_OK:
    st.error(
        "API credentials missing. Add OPENAI_API_KEY (and optionally OPENAI_API_BASE) to Streamlit secrets "
        "(.streamlit/secrets.toml), set them as environment variables, or provide a local config.json."
    )
    st.stop()

if not os.path.exists(DB_PATH):
    st.error(f"Database file '{DB_PATH}' not found. Place credit_risk_portfolio.db next to app.py "
             f"(or set the DB_PATH environment variable).")
    st.stop()

conn = get_connection()
ground_truth = load_ground_truth(TEST_QUERIES_PATH)

# --- Sidebar ------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.markdown(
        "- Recurring questions use **pre-approved SQL templates** (VQ1-VQ10)\n"
        "- Other questions get **generated, read-only SELECT** SQL\n"
        "- Every query passes a **5-check validation gate** (one retry), otherwise it is **escalated to an analyst**\n"
        "- Database access is **read-only**"
    )
    st.caption(f"Generation model: `{LLM_MODEL}`  |  Evaluator model: `{EVALUATOR_MODEL}`")

    st.header("Sample questions")
    sample_questions = ["How much of our book is in real estate?"]
    if ground_truth is not None and 'User Query' in ground_truth.columns:
        sample_questions += [q for q in ground_truth['User Query'].dropna().tolist() if q not in sample_questions]
    st.selectbox("Pick a sample question", sample_questions, key="sample_choice")

    def _use_sample():
        st.session_state["user_question"] = st.session_state["sample_choice"]

    st.button("Use this question", on_click=_use_sample)

# --- Portfolio snapshot (loan_master KPIs from the notebook) --------------------------
try:
    total_loans, outstanding, avg_rate, npa, npa_ratio = portfolio_summary(conn)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Loans", f"{total_loans:,}")
    k2.metric("Total Outstanding", f"${outstanding:,.2f} M")
    k3.metric("Avg. Interest Rate", f"{avg_rate:.2f}%")
    k4.metric("NPA Outstanding", f"${npa:,.2f} M")
    k5.metric("NPA Ratio", f"{npa_ratio:.2f}%")
except Exception as exc:  # noqa: BLE001
    st.warning(f"Portfolio snapshot unavailable: {exc}")

tab_ask, tab_audit, tab_eval, tab_library = st.tabs(
    ["Ask a question", "Audit trail", "Evaluation", "Verified query library"]
)

# --- Tab 1: Ask ---------------------------------------------------------------------
with tab_ask:
    st.text_area(
        "Enter your question:",
        key="user_question",
        height=100,
        placeholder="e.g., How much of our book is in real estate, and how much is non-performing?",
    )

    if st.button("Run Query Engine", type="primary"):
        question = st.session_state.get("user_question", "").strip()
        if not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Processing pipeline: classify -> build SQL -> validate -> execute -> respond..."):
                result, error = execute_with_audit(question, conn, source='interactive')
            if error is not None:
                st.session_state["last_result"] = None
                st.error(f"Pipeline Error: {error}")
            else:
                st.session_state["last_result"] = {"result": result, "question": question}

    if st.session_state.get("last_result"):
        render_result(st.session_state["last_result"]["result"], st.session_state["last_result"]["question"])

# --- Tab 2: Audit trail -------------------------------------------------------------
with tab_audit:
    st.subheader("Audit trail")
    st.caption("One record per analytical output: question, route, SQL, validation result, confidence, and narrative. "
               "Result rows are not stored.")
    records = load_audit_records()
    if not records:
        st.info("No queries have been run yet.")
    else:
        audit_df = pd.DataFrame(records)
        cols = ['timestamp_utc', 'source', 'status', 'user_question', 'route', 'query_id', 'confidence',
                'retry_used', 'escalated', 'row_count', 'executed_sql', 'narrative', 'error']
        audit_view = audit_df.reindex(columns=[c for c in cols if c in audit_df.columns]).astype(str)
        st.dataframe(audit_view.iloc[::-1])
        st.download_button(
            "Download full audit trail (JSONL)",
            data="\n".join(json.dumps(r, default=str) for r in records).encode("utf-8"),
            file_name="audit_log.jsonl",
            mime="application/json",
        )

# --- Tab 3: Evaluation --------------------------------------------------------------
with tab_eval:
    st.subheader("Evaluation against ground truth")
    required_cols = {'Test Case', 'User Query', 'Expected Route', 'Expected Query ID'}
    if ground_truth is None:
        st.info(f"Place `test_queries.csv` next to app.py (or set TEST_QUERIES_PATH) to enable evaluation. "
                f"Looked for: {TEST_QUERIES_PATH}")
    elif not required_cols.issubset(ground_truth.columns):
        st.error(f"test_queries.csv must contain the columns: {sorted(required_cols)}")
    else:
        st.dataframe(ground_truth)
        st.caption("Running the evaluation executes every test case through the full pipeline (LLM calls).")
        if st.button("Run evaluation"):
            st.session_state["evaluation"] = run_evaluation(ground_truth, conn)

        evaluation = st.session_state.get("evaluation")
        if evaluation:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Selected Path Accuracy", f"{evaluation['path_accuracy']:.1f}%")
            qa = evaluation['query_accuracy']
            m2.metric("Selected Query Accuracy", "N/A" if pd.isna(qa) else f"{qa:.1f}%")
            ac = evaluation['average_confidence']
            m3.metric("Average Confidence", "N/A" if pd.isna(ac) else f"{ac:.2f}")
            m4.metric("Escalation Rate", f"{evaluation['escalation_rate']:.1f}%")
            st.dataframe(evaluation['df'])
            st.markdown("**Validation confidence by test case**")
            st.bar_chart(evaluation['df'].set_index('Test Case')['Confidence'])

# --- Tab 4: Verified query library --------------------------------------------------
with tab_library:
    st.subheader("Verified query library")
    st.caption(f"{len(verified_query_library)} pre-approved, version-controlled SQL templates.")
    for qid, entry in verified_query_library.items():
        with st.expander(f"{qid}: {entry['description']}"):
            st.code(entry['sql'].strip(), language="sql")

    with st.expander("Database schema provided to the LLM"):
        st.code(database_schema.strip())

    with st.expander("Table previews (first 3 rows)"):
        for table in ['sector_master', 'loan_master', 'borrower_rating', 'provisioning']:
            st.markdown(f"**{table}**")
            try:
                st.dataframe(pd.read_sql_query(f"SELECT * FROM {table} LIMIT 3;", conn))
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not preview {table}: {exc}")

st.caption("Read-only analytical queries only. AI-generated SQL is validated before execution; "
           "complex or ambiguous questions are escalated to a human analyst.")
