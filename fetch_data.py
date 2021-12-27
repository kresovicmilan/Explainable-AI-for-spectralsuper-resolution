import requests
import json

# Fetch the example number 2
exampleIdx = 2
response = requests.get("http://172.17.0.2:6005/viz/api/set?iterator=" + str(exampleIdx))  # Reset the iterator to the first example
print("Example ID set response:", response.status_code)

# Fetch the data
response = requests.get("http://172.17.0.2:6005/viz/api/fetch")
print("Example fetch response:", response.status_code)
if response.status_code != 200:
    print("Error: Unable to retrieve data from service!")
    exit(-1)

data = json.loads(response.content.decode("utf-8"))["data"]

print("a")