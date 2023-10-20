import autogen
config_list = [
    {
        "api_type": "open_ai",
        "api_base": "http://localhost:1234/v1",
        "api_key": "sk-1111111111111111111111111"
    }
]
""" config_list = autogen.config_list_from_json(env_or_file="OAI_CONFIG_LIST") """
llm_config = {
    "seed": 42,
    "config_list": config_list,
}


user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin",
     code_execution_config={"last_n_messages": 2, "work_dir": "groopchat"},
    human_input_mode="TERMINATE")

coder = autogen.AssistantAgent("Coder", llm_config=llm_config)

user_proxy.initiate_chat(
    coder, message="Write python code to output number 1 to 122")

"""  from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})
user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding"})
user_proxy.initiate_chat(assistant, message="make me a python application that reads from a file and writes it to me")
# This initiates an automated chat between the two agents to solve the task 
import autogen
config_list = [
    {
        "api_type": "open_ai",
        "api_base": "http://localhost:1234/v1",
        "api_key": "NULL"
    }
]
llm_config = {
    # "request_timeout": 600,
    "seed": 42,
    "config_list": config_list,
    #  "temperature": 0
}

config_list = autogen.config_list_from_json(env_or_file="OAI_CONFIG_LIST")

coder = autogen.AssistantAgent("coder", llm_config=llm_config)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    system_message="A human admin",
    code_execution_config={"last_n_messages": 2, "work_dir": "groopchat"},
    human_input_mode="TERMINATE")


user_proxy.initiate_chat(
    coder, message="Write python code to output number 1 to 100")
 """
