import re
from litellm import completion
import json
import ast


def extract_answer(solution_str):
    match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    answer = extract_answer(solution_str)
    if answer is None:
        # print(f"DEBUG: Answer is None, solution_str: {solution_str[:200]}...")
        return 0
    else:
        if answer == str(ground_truth).strip():
            # print(f"DEBUG: {answer} == {str(ground_truth).strip()}")
            return score
        else:
            # print(f"DEBUG: {answer} != {str(ground_truth).strip()}")
            return format_score


def is_code_execution_successful(tool_output: str) -> bool:
    """
    Check if code execution was successful (no errors)
    Args:
        tool_output: The output result from code execution
    Returns:
        bool: True if execution was successful, False if there were errors
    """
    if not tool_output:
        return False
    
    tool_output = list(tool_output)
    if tool_output[0]['type'] == "error":
        return False
    
    return True


def clean_jupyter_output(raw_output, max_error_length: int = 800) -> str:
    """
    Clean and format Jupyter kernel output to be more readable.
    
    Args:
        raw_output: Raw output list/string from Jupyter kernel
        max_error_length: Maximum length for error messages (default: 800)
    
    Returns:
        str: Cleaned and formatted output
    """
    if not raw_output:
        return ""
    
    # Handle case where raw_output is already a list
    if isinstance(raw_output, list):
        output_list = raw_output
    else:
        # Try to parse string as Python literal (list/dict) or JSON
        try:
            try:
                output_list = ast.literal_eval(str(raw_output))
            except (ValueError, SyntaxError):
                try:
                    output_list = json.loads(str(raw_output))
                except json.JSONDecodeError:
                    # If both fail, return original with basic cleaning
                    return _clean_ansi_codes(str(raw_output))
        except Exception:
            return _clean_ansi_codes(str(raw_output))
    
    if not isinstance(output_list, list):
        return _clean_ansi_codes(str(output_list))
    
    cleaned_outputs = []
    
    for item in output_list:
        if not isinstance(item, dict):
            continue
            
        output_type = item.get('type', '')
        
        if output_type == 'result':
            # Handle successful execution result
            data = item.get('data', {})
            if isinstance(data, dict):
                # Prefer text/plain over text/html
                if 'text/plain' in data:
                    content = data['text/plain']
                    cleaned_outputs.append(content)
                elif 'text/html' in data:
                    # Strip HTML tags as fallback
                    html_content = data['text/html']
                    cleaned_content = _strip_html_tags(html_content)
                    cleaned_outputs.append(cleaned_content)
                else:
                    # Use any other text content available
                    for key, value in data.items():
                        if isinstance(value, str):
                            cleaned_outputs.append(f"{key}: {value}")
                            
        elif output_type == 'error':
            # Handle error output
            error_name = item.get('name', 'Error')
            error_value = item.get('value', '')
            traceback = item.get('traceback', [])
            
            # Clean error message
            error_msg = f"{error_name}: {error_value}"
            
            # Clean and truncate traceback
            if isinstance(traceback, list):
                # Remove ANSI codes from traceback lines
                clean_traceback = []
                for line in traceback:
                    clean_line = _clean_ansi_codes(str(line))
                    clean_traceback.append(clean_line)
                
                # Join and truncate if too long
                traceback_str = '\n'.join(clean_traceback)
                if len(traceback_str) > max_error_length:
                    traceback_str = traceback_str[:max_error_length] + "\n... (truncated)"
                
                error_msg = f"{error_msg}\n{traceback_str}"
            
            cleaned_outputs.append(error_msg)
            
        elif output_type == 'stream':
            # Handle stream output (stdout, stderr)
            name = item.get('name', '')
            text = item.get('text', '')
            if text:
                clean_text = _clean_ansi_codes(text)
                cleaned_outputs.append(f"[{name}] {clean_text}" if name else clean_text)
    
    return '\n'.join(cleaned_outputs) if cleaned_outputs else ""


def _clean_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from text."""
    if not text:
        return ""
    # Remove ANSI escape sequences
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)


def _strip_html_tags(html: str) -> str:
    """Strip HTML tags and extract plain text."""
    if not html:
        return ""
    # Remove HTML tags
    clean = re.compile('<.*?>')
    text = clean.sub('', html)
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_llm_score(solution_str, history_str, ground_truth, query, method="strict", format_score=0.0, score=1.0):
    '''
Entire Trajectory:
{history_str}
    '''
    if solution_str is None:
        return 0.0
    solution_str =  str(solution_str)
    ground_truth = str(ground_truth)
    history_str = str(history_str)

    prompt = f"""You are a judge evaluating scientific hypotheses. You need to score how well the predicted hypothesis matches the ground truth hypothesis.
Both the hypotheses answer the natural language query "Query" over the dataset(s).
To evaluate the hypothesis, you need to consider three dimensions that define the hypothesis: Context, Variables, and Relations. 
Here are the definitions for these dimensions:
- Contexts: Boundary conditions that limit the scope of a hypothesis. E.g., “for men over \
the age of 30”, “in Asia and Europe”. 
- Variables: Known concepts that interact in a meaningful way under a given context to \
produce the hypothesis. E.g., gender, age, income, or "None" if there is no interacting variable.
- Relations: Interactions between a given set of variables under a given context to produce \
the hypothesis. E.g., “quadratic relationship”, “inversely proportional”, piecewise conditionals, \
or "None" if there is no interacting relationship.
Compare the predicted hypothesis with the ground truth hypothesis in terms of these three dimensions.

Query:
{query}

Predicted Hypothesis:
{solution_str}

Ground Truth Hypothesis:
{ground_truth}

Evaluate the hypothesis and provide a score between 0 and 1, where:
- 1.0 means the hypotheses make the same scientific claim
- 0.0 means completely different or contradictory claims
- Values between 0 and 1 indicate partial alignment in variables, relationships, or context
- If the predicted hypothesis is None, return 0.0

Return your response in the following format:
<reasoning>REASONING</reasoning>
<answer>SCORE</answer>

Only return the numeric score between 0 and 1 within the answer tags."""

    try:
        response = completion(
            model="together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            # model="o4-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        score_str = extract_answer(response.choices[0].message.content)
        if score_str:
            score = float(score_str)
            return max(0.0, min(1.0, score))
        return 0.0
    except Exception:
        return 0.0

def compute_llm_score_discrete(solution_str, history_str, ground_truth, query, method="strict", format_score=0.0, score=1.0):
    if solution_str is None:
        return 0.0
    solution_str =  str(solution_str)
    ground_truth = str(ground_truth)
    history_str = str(history_str)

    prompt = f"""You are an impartial judge. Decide if the model's final hypothesis to the query is correct.
Follow these dataset-specific rules for DiscoveryBench:
- If the answer is numerical, treat it as correct if the relative error < 1% compared with the ground-truth value.
- Otherwise, judge correctness against the provided ground-truth answer.

Query: {query}

Predicted hypothesis: {solution_str}

Ground-truth hypothesis: {ground_truth}


Return your response in the following format:
<reasoning>REASONING</reasoning>
<answer>SCORE</answer>

Only return 0 or 1 within the answer tags."""

    try:
        response = completion(
            model="together_ai/deepseek-ai/DeepSeek-V3",
            # model="o4-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        score_str = extract_answer(response.choices[0].message.content)
        if score_str:
            score = float(score_str)
            return max(0.0, min(1.0, score))
        return 0.0
    except Exception:
        return 0.0

def compute_llm_score_discrete_multi(solution_str, history_str, ground_truth, query, method="strict", format_score=0.0, score=1.0):
    if solution_str is None:
        return 0.0
    solution_str =  str(solution_str)
    ground_truth = str(ground_truth)
    history_str = str(history_str)

    prompt = f"""You are a judge evaluating scientific hypotheses. You need to score how well the predicted hypothesis matches the ground truth hypothesis.
Both the hypotheses answer the natural language query "Query" over the dataset(s).
To evaluate the hypothesis, you need to consider three dimensions that define the hypothesis: Context, Variables, and Relations. 
Here are the definitions for these dimensions:
- Contexts: Boundary conditions that limit the scope of a hypothesis. E.g., “for men over \
the age of 30”, “in Asia and Europe”. 
- Variables: Known concepts that interact in a meaningful way under a given context to \
produce the hypothesis. E.g., gender, age, income, or "None" if there is no interacting variable.
- Relations: Interactions between a given set of variables under a given context to produce \
the hypothesis. E.g., “quadratic relationship”, “inversely proportional”, piecewise conditionals, \
or "None" if there is no interacting relationship.

Compare the predicted hypothesis with the ground truth hypothesis in terms of these three dimensions.

Query:
{query}

Predicted Hypothesis:
{solution_str}

Ground Truth Hypothesis:
{ground_truth}

Evaluate the hypothesis and provide your score from the following options, where:
- 0: Two hypotheses are completely different or contradict each other; Major context mismatch, or variables/relations do not align at all.
- 0.2: The two hypotheses are weakly aligned. Context unclear OR minor context mismatch; some overlap/compatibility in variables/relations.
- 0.5: The two hypotheses are partially aligned. Context equivalent; variables/relations are partially aligned; some variables/relations are incompatible but not contradictory.
- 0.8: There are no contradictions and the two hypotheses are highly aligned. Context equivalent and variables and relation mostly match with only minor omissions or weaker phrasing.
- 1.0: Two hypotheses make the same scientific claim. Context equivalent AND variables match/superset AND relation matches.

Return your response in the following format:
<reasoning>REASONING</reasoning>
<answer>SCORE</answer>

Only return the numeric score chosen from the options (0, 0.2, 0.5, 0.8, 1.0) within the answer tags."""

    try:
        response = completion(
            model="together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            # model="o4-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        score_str = extract_answer(response.choices[0].message.content)
        if score_str:
            score = float(score_str)
            return max(0.0, min(1.0, score))
        return 0.0
    except Exception:
        return 0.0

if __name__ == "__main__":
    # solution = """def add(a, b):
    # return a + b"""
    # ground_truth = """def add(a, b):
    # return a + b"""
    
    # score = compute_llm_score(solution, ground_truth)
    # print(f"Score: {score}") 
    raw_output = [{'type': 'error', 'name': 'FileNotFoundError', 'value': "[Errno 2] No such file or directory: '/data/qrdata/data/Neuropic_36.csv'", 'traceback': ['\x1b[31m---------------------------------------------------------------------------\x1b[39m', '\x1b[31mFileNotFoundError\x1b[39m                         Traceback (most recent call last)', '\x1b[36mCell\x1b[39m\x1b[36m \x1b[39m\x1b[32mIn[1]\x1b[39m\x1b[32m, line 3\x1b[39m\n\x1b[32m      1\x1b[39m \x1b[38;5;28;01mimport\x1b[39;00m\x1b[38;5;250m \x1b[39m\x1b[34;01mpandas\x1b[39;00m\x1b[38;5;250m \x1b[39m\x1b[38;5;28;01mas\x1b[39;00m\x1b[38;5;250m \x1b[39m\x1b[34;01mpd\x1b[39;00m\n\x1b[32m      2\x1b[39m file_path = \x1b[33m"\x1b[39m\x1b[33m/data/qrdata/data/Neuropic_36.csv\x1b[39m\x1b[33m"\x1b[39m\n\x1b[32m----> \x1b[39m\x1b[32m3\x1b[39m data = \x1b[43mpd\x1b[49m\x1b[43m.\x1b[49m\x1b[43mread_csv\x1b[49m\x1b[43m(\x1b[49m\x1b[43mfile_path\x1b[49m\x1b[43m)\x1b[49m\n\x1b[32m      4\x1b[39m data.head(\x1b[32m10\x1b[39m)\n', '\x1b[36mFile \x1b[39m\x1b[32m/usr/local/lib/python3.12/site-packages/pandas/io/parsers/readers.py:1026\x1b[39m, in \x1b[36mread_csv\x1b[39m\x1b[34m(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)\x1b[39m\n\x1b[32m   1013\x1b[39m kwds_defaults = _refine_defaults_read(\n\x1b[32m   1014\x1b[39m     dialect,\n\x1b[32m   1015\x1b[39m     delimiter,\n\x1b[32m   (...)\x1b[39m\x1b[32m   1022\x1b[39m     dtype_backend=dtype_backend,\n\x1b[32m   1023\x1b[39m )\n\x1b[32m   1024\x1b[39m kwds.update(kwds_defaults)\n\x1b[32m-> \x1b[39m\x1b[32m1026\x1b[39m \x1b[38;5;28;01mreturn\x1b[39;00m \x1b[43m_read\x1b[49m\x1b[43m(\x1b[49m\x1b[43mfilepath_or_buffer\x1b[49m\x1b[43m,\x1b[49m\x1b[43m \x1b[49m\x1b[43mkwds\x1b[49m\x1b[43m)\x1b[49m\n', '\x1b[36mFile \x1b[39m\x1b[32m/usr/local/lib/python3.12/site-packages/pandas/io/parsers/readers.py:620\x1b[39m, in \x1b[36m_read\x1b[39m\x1b[34m(filepath_or_buffer, kwds)\x1b[39m\n\x1b[32m    617\x1b[39m _validate_names(kwds.get(\x1b[33m"\x1b[39m\x1b[33mnames\x1b[39m\x1b[33m"\x1b[39m, \x1b[38;5;28;01mNone\x1b[39;00m))\n\x1b[32m    619\x1b[39m \x1b[38;5;66;03m# Create the parser.\x1b[39;00m\n\x1b[32m--> \x1b[39m\x1b[32m620\x1b[39m parser = \x1b[43mTextFileReader\x1b[49m\x1b[43m(\x1b[49m\x1b[43mfilepath_or_buffer\x1b[49m\x1b[43m,\x1b[49m\x1b[43m \x1b[49m\x1b[43m*\x1b[49m\x1b[43m*\x1b[49m\x1b[43mkwds\x1b[49m\x1b[43m)\x1b[49m\n\x1b[32m    622\x1b[39m \x1b[38;5;28;01mif\x1b[39;00m chunksize \x1b[38;5;129;01mor\x1b[39;00m iterator:\n\x1b[32m    623\x1b[39m     \x1b[38;5;28;01mreturn\x1b[39;00m parser\n', '\x1b[36mFile \x1b[39m\x1b[32m/usr/local/lib/python3.12/site-packages/pandas/io/parsers/readers.py:1620\x1b[39m, in \x1b[36mTextFileReader.__init__\x1b[39m\x1b[34m(self, f, engine, **kwds)\x1b[39m\n\x1b[32m   1617\x1b[39m     \x1b[38;5;28mself\x1b[39m.options[\x1b[33m"\x1b[39m\x1b[33mhas_index_names\x1b[39m\x1b[33m"\x1b[39m] = kwds[\x1b[33m"\x1b[39m\x1b[33mhas_index_names\x1b[39m\x1b[33m"\x1b[39m]\n\x1b[32m   1619\x1b[39m \x1b[38;5;28mself\x1b[39m.handles: IOHandles | \x1b[38;5;28;01mNone\x1b[39;00m = \x1b[38;5;28;01mNone\x1b[39;00m\n\x1b[32m-> \x1b[39m\x1b[32m1620\x1b[39m \x1b[38;5;28mself\x1b[39m._engine = \x1b[38;5;28;43mself\x1b[39;49m\x1b[43m.\x1b[49m\x1b[43m_make_engine\x1b[49m\x1b[43m(\x1b[49m\x1b[43mf\x1b[49m\x1b[43m,\x1b[49m\x1b[43m \x1b[49m\x1b[38;5;28;43mself\x1b[39;49m\x1b[43m.\x1b[49m\x1b[43mengine\x1b[49m\x1b[43m)\x1b[49m\n', '\x1b[36mFile \x1b[39m\x1b[32m/usr/local/lib/python3.12/site-packages/pandas/io/parsers/readers.py:1880\x1b[39m, in \x1b[36mTextFileReader._make_engine\x1b[39']}]

    clean_output = clean_jupyter_output(raw_output)
    print(f"{clean_output}")