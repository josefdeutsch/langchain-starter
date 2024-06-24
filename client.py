from langserve import RemoteRunnable
from pprint import pprint

remote_chain = RemoteRunnable("http://localhost:8000/chain/")
response = remote_chain.invoke({"language": "italian", "text": "hi"})

pprint(response)