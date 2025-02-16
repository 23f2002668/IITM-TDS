import os, subprocess, json, datetime, openai, requests, base64, numpy as np, sqlite3
from flask import Flask, jsonify, request, abort
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Configure OPENAI API Key
api_key = os.environ.get("AIPROXY_TOKEN")

if not api_key:
    raise ValueError("AIPROXY_TOKEN environment variable is not set.")

user_email = "23f2002668@ds.study.iitm.ac.in"



# ------------------------------------------------------------------------------------------------------ #


@app.route("/", methods=["GET"])
def home():
    return "Server is running!", 200



# ------------------------------------------------------------------------------------------------------ #



def validate_path(path):
    data_dir = Path("data").resolve()
    resolved_path = Path(path).resolve()

    if not resolved_path.is_relative_to(data_dir):
        raise ValueError("Access to files outside /data is not allowed.")
    if not resolved_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(resolved_path)
    return True


# ------------------------------------------------------------------------------------------------------ #


# The first commands to be executed
def first(input_file, pyfile):
    try:
        # Run subprocess commands and check for errors
        subprocess.run("pip3 install uv", shell=True, check=True, timeout=60)  # Timeout added
        subprocess.run(f"curl -O h{input_file}", shell=True, check=True, timeout=60)
        subprocess.run(f"python3 {pyfile} {user_email} --root ./data", shell=True, check=True, timeout=60)
    except subprocess.CalledProcessError as e:
        print(f"Error in subprocess: {e}")
        return f"Error occurred during subprocess execution: {e}"
    except subprocess.TimeoutExpired as e:
        print(f"Subprocess timed out: {e}")
        return "Subprocess timed out"

    return "Subprocess commands run successfully!"


# ------------------------------------------------------------------------------------------------------ #


# Get the steps of tasks from the LLM
def parse_task(task, api_key):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "gpt-4o-mini",
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": f"Parse the following task description into structured JSON format strictly with same keys as mentioned : 1. 'inputFilePath', 2. 'operation', 3. 'outputFilePath'. If there is no inputFilePath or outputFilePath mentioned in task then the value of it should be 'No'. If the operation is to run url of python not the github repo clone then extract last python file name (xyz.py) from link and then input file have value the entire url link and operation just have a list with 'run' at 0th index and then file name like ['run', 'xyz.py']. If the operation is to format the content using prettier the the operation should have value a list with 'format' at 0th index and then 'prettier' like ['format', 'prettier']. If the operation is performed on weekdays like tuesday or etc., then the operation should have value a list with 0th index 'day' and then the operated day like ['day' , 'Tuesday']. If the operation is sorting like sorting contacts, etc., then the value of 'operation' should be a list with operation to be performed 'sort' at 0th index which is given as ['sort', 'first name', 'last name', and so on..]. If the opeartion is to sort 'most frequent', 'least frequent' or 'most recent' or 'least recent' n log files then the value of 'operation' should be a list with operation to be performed 'sort log' at 0th index which is given as ['sort log', 'n', 'least frequent']. If the operation has to perform on markdown file then the operation should have value a list with 0th index 'md' like ['md', '#', 'H1']. If the operation is about extracting information such as reciever mobile number then the value of 'operation' should be a list with operation to be performed 'extract' at 0th index which is given as ['extract', 'extract the reciever mobile number']. If the operation is to extract some information like mobile number from image then the value of operation should be a list with 'image' at 0th index and extracting information 'mobile number' after it like ['image', 'mobile number']. If the operation requires or use embeddings then operation should have a string value which is 'embedding'. If the operation have to perform on database like 'The SQLite database file /data/ticket-sales.db has a tickets with columns type, units, and price. What is the total sales of all the items in the “Silver” ticket type?' then the operation should have value list with 'database' at 0th index and then table name, item like ['database', 'tickets', 'Silver']. If the operation is to delete then the value of operation should be a list with 'delete' at 0th index like ['delete']. Here The task is given as '{task}'"
            },
            {
                "role": "user",
                "content": task
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    result = response.json()
    result = result["choices"][0]["message"]["content"]
    result = json.loads(result)

    return [result["inputFilePath"], result["operation"], result["outputFilePath"]]




# ------------------------------------------------------------------------------------------------------ #




def parse_task_B(task, api_key):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a task parser. Parse the following task description into structured JSON format. "
                    "Output only Python programs or Bash commands without any extra keywords. "
                    "The broad additional tasks are listed for automation. But the tasks are not defined more precisely."
                    "You must able to handle these tasks as well come up with that are outside of this list."
                    "The tasks could be as follows :"
                    "B1. Fetch data from an API and save it."
                    "B2. Clone a git repo and make a commit"
                    "B3. Run a SQL query on a SQLite or DuckDB database"
                    "B4. Extract data from (i.e. scrape) a website"
                    "B5. Compress or resize an image"
                    "B6. Transcribe audio from an MP3 file"
                    "B7. Convert Markdown to HTML"
                    "B8. Write an API endpoint that filters a CSV file and returns JSON data"
                    "Use the following schema:\n"
                    "{\n"
                    "  \"application_type\": \"PYTHON/BASH\",\n"
                    "  \"task_code\": \"code as string\"\n"
                    "}"
                )
            },
            {
                "role": "user",
                "content": task
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        try:
            result_content = result["choices"][0]["message"]["content"]
            parsed_result = json.loads(result_content)
            return parsed_result
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error parsing response: {e}")
            return None
    else:
        print(f"Request failed with status code: {response.status_code}")
        return None





# ------------------------------------------------------------------------------------------------------ #




# Function for getting weekday
def week_day(date):

    date_formats = [
        "%b %d, %Y",  # "Aug 16, 2011"
        "%d-%b-%Y",  # "07-Aug-2011"
        "%Y-%m-%d",  # "2020-07-22"
        "%Y/%m/%d %H:%M:%S",  # "2002/07/28 02:13:43"
        "%d-%b-%Y",  # "25-Nov-2015"
        "%Y/%m/%d %H:%M:%S",  # "2016/01/26 09:04:02"
        "%Y/%m/%d",  # "2016/01/26"
        "%Y-%m-%d",  # "2001-04-30"
        "%d-%b-%Y",  # "16-Mar-2000"
    ]

    for fmt in date_formats:
        try:
            date = datetime.datetime.strptime(date, fmt) # It convert all dates into '2011-08-16 00:00:00' format
            return date.strftime("%A")

        except ValueError:
            continue  # If it doesn't match this format, try the next one

    return "Invalid Date Format"  # If none of the formats match, return this message



# ------------------------------------------------------------------------------------------------------ #



def recent_logs(input_file, operation, output_file):

    n, most = operation[1], operation[2]

    # Get all .log files in /data/logs/, sorted by modification time (most recent first)
    log_dir = Path(input_file)

    if most == "most recent" or most == "most frequent":
        log_files = sorted(log_dir.glob('*.log'), key=os.path.getmtime, reverse=True)[:10]
    else:
        log_files = sorted(log_dir.glob('*.log'), key=os.path.getmtime)[:10]

    recent_lines = []
    for file in log_files:
        with open(file, 'r') as f:
            first_line = f.readline().strip()  # Read and strip any trailing newline
            recent_lines.append(first_line)

    with open(output_file, 'w') as f:
        for line in recent_lines:
            f.write(line + '\n')

    print("Operation on logs files has completed !")




# ------------------------------------------------------------------------------------------------------ #




def markdown(input_file):

    docs_dir = Path(input_file)

    index = {}

    md_files = docs_dir.rglob('*.md')

    for md_file in md_files:
        with open(md_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('# '):  # Check for H1 (single #)
                    # Get the filename without the prefix
                    relative_filename = str(md_file.relative_to(docs_dir))
                    # Extract the title text after '# '
                    title = line[2:].strip()
                    index[relative_filename] = title
                    break  # Only the first H1 is needed

    index_file = docs_dir / 'index.json'
    with open(index_file, 'w') as f:
        json.dump(index, f)

    print("Index has been created at /data/docs/index.json")




# ------------------------------------------------------------------------------------------------------ #



def email(r, operation):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an extraction engine. Extract only the required information without any extra text. "
                    "Respond concisely and precisely."
                )
            },
            {
                "role": "user",
                "content": f"{operation} from the data: {r}"
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        print("Request failed with status code:", response.status_code)
        print("Response:", response.text)
        return None

    result = response.json()

    extracted_info = result["choices"][0]["message"]["content"].strip()

    return extracted_info




# ------------------------------------------------------------------------------------------------------ #




def image(input_file, operation):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Read the image file as binary data
    with open(input_file, 'rb') as image_file:
        image_data = image_file.read()

    # Encode the image data to base64
    base64_image = base64.b64encode(image_data).decode('utf-8')

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Extract the {operation} without any extra text."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "detail": "low",
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    }

    response = requests.post("https://aiproxy.sanand.workers.dev/openai/v1/chat/completions", headers=headers,
                             json=data)

    if response.status_code == 200:
        response_json = response.json()
        print(response_json["choices"][0]["message"]["content"])
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)




# ------------------------------------------------------------------------------------------------------ #




def get_embedding(text):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are an encoder that generates numerical embeddings for text."
            },
            {
                "role": "user",
                "content": f"Generate a numerical embedding for this text: '{text}'"
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        print("Request failed with status code:", response.status_code)
        print("Response:", response.text)
        return None

    result = response.json()

    extracted_info = result["choices"][0]["message"]["content"].strip()

    return extracted_info


def cosine_similarity(vec1, vec2):
    """ Calculate cosine similarity between two vectors. """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def find_most_similar_comments(input_file, output_file):
    # Step 1: Read comments from file
    with open(input_file, "r") as f:
        comments = [line.strip() for line in f.readlines()]

    # Step 2: Generate embeddings for each comment
    embeddings = []
    for comment in comments:
        embedding = get_embedding(comment)
        if embedding is not None:
            embeddings.append(embedding)
        else:
            print(f"Failed to get embedding for: {comment}")

    # Step 3: Calculate similarities
    max_similarity = -1
    most_similar_pair = ("", "")
    for i in range(len(comments)):
        for j in range(i + 1, len(comments)):
            similarity = cosine_similarity(embeddings[i], embeddings[j])
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_pair = (comments[i], comments[j])

    # Step 4: Write the most similar pair to the output file
    with open(output_file, "w") as f:
        f.write(most_similar_pair[0] + "\n")
        f.write(most_similar_pair[1] + "\n")

    print(f"Most similar comments saved to {output_file}")




# ------------------------------------------------------------------------------------------------------ #




def calculate_gold_sales(db_file, output_file, item, table):
    # Step 1: Connect to the database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    query = f"""
        SELECT SUM(units * price) 
        FROM {table} 
        WHERE type = ?;
    """
    cursor.execute(query, (item,))
    total_sales = cursor.fetchone()[0]

    if total_sales is None:
        total_sales = 0

    with open(output_file, 'w') as f:
        f.write(str(total_sales))

    cursor.close()
    conn.close()

    print(f"Total sales for {item} tickets saved to {output_file}")




# ------------------------------------------------------------------------------------------------------ #




# Execute the steps
def execute_step(input_file, operation, output_file):

    if 'run' in operation:
        pyfile = operation[1]
        r = first(input_file, pyfile)
        print("The python file has downloaded !")



    elif "format" in operation:
        print(f"The {input_file} is formatting .....")
        subprocess.run("npm install prettier@3.4.2", shell=True, timeout=60)
        subprocess.run(f"npx prettier --write {input_file}", shell=True, timeout=60)
        print(f"The {input_file} has formatted with Prettier@3.4.2 !")



    elif "day" in operation or "date" in operation:
        subprocess.run("pip install datetime", shell=True)
        count, req_day = 0, "Sunday"

        if "sunday" in operation:
            req_day = "Sunday"
        elif "monday" in operation:
            req_day = "Monday"
        elif "tuesday" in operation:
            req_day = "Tuesday"
        elif "wednesday" in operation:
            req_day = "Wednesday"
        elif "thursday" in operation:
            req_day = "Thursday"
        elif "friday" in operation:
            req_day = "Friday"
        else:
            req_day = "Saturday"


        with open(input_file, 'r') as f:
            for date in f:
                date = date.strip()
                day = week_day(date)
                print(f"{date}  ===>  {day}")
                if day == req_day:
                    count += 1

        with open(output_file, 'w') as f:
            f.write(str(count))

        print("Operation on wekdays has completed !")



    elif "sort" in operation:
        with open(input_file, 'r') as f:
            contacts = json.load(f)

        op = operation[1:]

        sorted_contacts = sorted(contacts, key=lambda x: tuple(x[field] for field in op))

        with open(output_file, 'w') as f2:
            json.dump(sorted_contacts, f2)

        print("Sorted operation has completed !")



    elif "log" in operation:
        recent = recent_logs(input_file, operation, output_file)
        print(recent)



    elif "H1" in operation or "md" in operation or "#" in operation:
        mark = markdown(input_file)
        print(mark)




    elif "extract" in operation or "extract" in operation[1]:
        f = open(input_file, "r")
        r = f.read()
        result = email(r, operation)

        f2 = open(output_file, 'w')
        f2.write(result)
        f.close()
        f2.close()

        print("The required information has been extracted and written !")




    elif "image" in operation:
            result = image(input_file, operation)
            result = result.replace(" ", "")
            f = open(output_file, 'w')
            f.write(result)
            f.close()




    elif 'embedding' in operation or 'embeddings' in operation:
        emb = find_most_similar_comments(input_file, output_file)
        print(emb)




    elif "database" in operation:
            table, item = operation[1], operation[2]
            data = calculate_gold_sales(input_file, output_file, item, table)
            print(data)



    else:
        return 'Completed !'




# ------------------------------------------------------------------------------------------------------ #


@app.route("/run", methods=["POST"])
def run_task():
    task = request.args.get('task')
    if not task:
        abort(400, description="Task description is required")
    print(f"Task : {task}")

    try:
        l = parse_task(task, api_key)
        input_file, operation, output_file = l[0], l[1], l[2]
        input_file, output_file = input_file[1:], output_file[1:]

        print(f"Task  :   {l} \nInput File   :   {input_file} \nOperation   :   {operation} \nOutput File   :   {output_file}")

        if ".py" not in input_file and input_file != "o":
            iv = validate_path(input_file)
            print(iv)
        else:
            iv = True

        if iv :
            if 'delete' not in operation:
                response = execute_step(input_file, operation, output_file)
                print(response)

                if response == 'Completed !':
                    response = parse_task_B(task, api_key)

                    if response['application_type'] == "BASH":
                        steps = response['task_code']
                        subprocess.run(steps, shell=True, capture_output=True)
                        print("Subprocess commands run successfully !")

                    elif response['application_type'] == "PYTHON":
                        steps = response['task_code']
                        print(steps)
                        exec(steps)
                        print("Subprocess commands run successfully !")

                return jsonify({"status": "success"}), 200

            else:
                return jsonify({"status": "success", "message" : "Sorry ! Data cannot be deleted in any codition."}), 200

        else:
            return jsonify({"status": "success", "message" : "Data outside /data is never accessed or exfiltrated !"}), 200

    except Exception as e:
        abort(500, description=str(e))



# ------------------------------------------------------------------------------------------------------ #



@app.route("/read", methods=["GET"])
def read_file():
    file_path = request.args.get("path")
    file_path = file_path[1:]

    v = validate_path(file_path)

    if v:
        if not file_path:
            abort(400, description="File path is required")
        if not os.path.exists(file_path):
            abort(404, description="File not found")
        with open(file_path, 'r') as f:
            content = f.read()
        return content, 200

    else:
        return "Data outside /data is never accessed or exfiltrated !"




# ------------------------------------------------------------------------------------------------------ #



if __name__ == "__main__":
    app.debug=True
    app.run(host='0.0.0.0', port=8000, use_reloader=False)


"""
curl -X POST "http://localhost:8000/run?task=Format%20/data/format.md%20with%20prettier%203.4.2"

Build Docker Image And Run It
docker build --no-cache -t dataworks-agent .
docker run -e AIPROXY_TOKEN=$AIPROXY_TOKEN -p 8000:8000 dataworks-agent


# Test the API

curl -X POST "http://localhost:8000/run?task=Install+uv+(if+required)+and+run+https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py+with+${user.email}+as+the+only+argument.+(NOTE:+This+will+generate+data+files+required+for+the+next+tasks.)"
"""
