# Challenge for Synthetic Request
import openai
import random
from protocol import LogicSynapse

import bittensor as bt
from human_noise import get_condition, get_sample_condition
from math_generator.topics import TOPICS as topics
import mathgenerator
import time
from rewarder import LogicRewarder

base_url = "http://localhost:8000/v1"
client_model = "Qwen/Qwen2-72B-Instruct"
client_key = "xyz"

openai_client = openai.OpenAI(base_url=base_url, api_key=client_key)


sample_synapse = LogicSynapse(timeout=64)
sample_synapse.raw_logic_question = "Find the solution of this math problem:\n---\nTopic: Algebra, Subtopic: Complex quadratic.\nFind the roots of given Quadratic Equation 2x^2 + 6x + 2 = 0\n---"
sample_synapse.ground_truth_answer = "((-0.382, -2.618)) = (\\frac{-6 + \sqrt{20}}{2*2}, (\\frac{-6 - \sqrt{20}}{2*2})"

sample_topic = "Algebra"
sample_subtopic = "Complex quadratic"


sample_conditions = {
        "profile": "math hobbyist",
        "mood": "curious",
        "tone": "playful",
    }


def solve(synapse: LogicSynapse):
    logic_question: str = synapse.logic_question
    messages = [
            {"role": "user", "content": logic_question},
        ]
    response = openai_client.chat.completions.create(
        model=client_model,
        messages=messages,
        max_tokens=2048,
        temperature=0.8,
    )
    synapse.logic_reasoning = response.choices[0].message.content

    messages.extend(
        [
            {"role": "assistant", "content": synapse.logic_reasoning},
            {
                "role": "user",
                "content": "Give me the final short answer as a sentence. Don't reasoning anymore, just say the final answer in the same math latex. \
                    For example, given the problem 'Find the roots of given Quadratic Equation $x^2 + 8x + 8 = 0$', the answer is $(-1.172, -6.828) = (\frac{-8 + \sqrt{32}}{2}, (\frac{-8 - \sqrt{32}}{2})$",
            },
        ]
    )

    response = openai_client.chat.completions.create(
        model=client_model,
        messages=messages,
        max_tokens=512,
        temperature=0.7,
    )
    synapse.logic_answer = response.choices[0].message.content
    # synapse.logic_answer = "((-0.382, -2.618)) = (\frac{-6 + \sqrt{20}}{2*2}, (\frac{-6 - \sqrt{20}}{2*2})"
    print("\n")
    print("="*50 + "logic_answer" + "="*50)
    print(synapse.logic_answer)
    print("="*120)
    print("\n")
    # print(f"Logic answer: {synapse.logic_answer}")
    # print(f"Logic reasoning: {synapse.logic_reasoning}")
    return synapse


class LogicChallenger:
    def __init__(self, base_url: str, api_key: str, model: str):
        bt.logging.info(
            f"Logic Challenger initialized with model: {model}, base_url: {base_url}"
        )
        self.model = model
        self.openai_client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def __call__(self, synapse: LogicSynapse) -> LogicSynapse:
        self.get_challenge(synapse)
        # self.get_sample_challenge(synapse)
        return synapse

    def get_sample_challenge(self, synapse: LogicSynapse):
        logic_problem = self.get_sample_math_problem(synapse)
        conditions: dict = get_sample_condition()
        print(f"***********************Human Condition: {conditions}******************\n")
        revised_logic_question: str = self.get_revised_math_question(
            logic_problem, conditions
        )
        synapse.logic_question = revised_logic_question
        print("="*50 + "logic_question" + "="*50)
        print(synapse.logic_question)
        print("="*120)

    def get_challenge(self, synapse: LogicSynapse):
        logic_problem = self.get_atom_math_problem(synapse)
        # conditions: dict = get_condition()

        conditions = sample_conditions

        
        print(f"***********************Human Condition: {conditions}******************\n")
        revised_logic_question: str = self.get_revised_math_question(
            logic_problem, conditions
        )
        revised_logic_question = "Hey there fellow math enthusiasts! Let's dive into the crazy world of quadratic equations, shall we? I've stumbled upon this little beauty: 2x² + 6x + 2 = 0. Can you help me find its roots? Remember, we're looking for the values of x that make this equation true. So, let's get our thinking caps on and see if we can crack this one together! Don't forget to show your work, because I'm curious to see if we can do this the old-fashioned way, without a calculator. Let's go, team!"
        synapse.logic_question = revised_logic_question
        print("="*50 + "logic_question" + "="*50)
        print(synapse.logic_question)
        print("="*120)

    def get_sample_math_problem(self, synapse: LogicSynapse) -> str:
        subtopic = "complex_to_polar"
        topic = "misc"
        sample_problem, sample_answer = eval(f"mathgenerator.{topic}.{subtopic}()")
        sample_problem = sample_problem.replace("$", "").strip()
        sample_problem = f"Find the solution of this math problem:\n---\nTopic: {topic}, Subtopic: {subtopic}.\n{sample_problem}\n---\n"
        
        synapse.raw_logic_question = sample_problem
        print("Raw logic problem\n-------------------------------------------------------------")
        print(sample_problem)
        print("------------------------------------------------------------------")

        synapse.ground_truth_answer = str(sample_answer).replace("$", "").strip()
        
        return sample_problem

    def get_atom_math_problem(self, synapse: LogicSynapse) -> str:
        selected_topic = random.choice(topics)
        subtopic = selected_topic["subtopic"]
        topic = selected_topic["topic"]
        
        # atom_problem, atom_answer = eval(f"mathgenerator.{topic}.{subtopic}()")
        subtopic = subtopic.replace("_", " ").capitalize()
        topic = topic.replace("_", " ").capitalize()
        
        # atom_problem = atom_problem.replace("$", "").strip()
        # atom_problem = f"Find the solution of this math problem:\n---\nTopic: {topic}, Subtopic: {subtopic}.\n{atom_problem}\n---\n"
        
        # synapse.raw_logic_question = atom_problem
        # synapse.ground_truth_answer = str(atom_answer).replace("$", "").strip()

        synapse.raw_logic_question = sample_synapse.raw_logic_question
        synapse.ground_truth_answer = sample_synapse.ground_truth_answer
        atom_problem = synapse.raw_logic_question

        print(f"******************topic : {topic}************subtopic : {subtopic}*******************\n")
        print("="*50 + "raw_logic_question" + "="*50)
        print(synapse.raw_logic_question)
        print("="*120)
        print("\n")
        print("="*50 + "ground truth answer" + "="*50)
        print(synapse.ground_truth_answer)
        print("="*120)
        
        return atom_problem

    def get_revised_math_question(self, math_problem: str, conditions: dict) -> str:
        prompt = "Please paraphrase by adding word or expression to this question as if you were a {profile} who is {mood} and write in a {tone} tone. You can use incorrect grammar, typo or add more context! Don't add your solution! Just say the revised version, you don't need to be polite.".format(
            **conditions
        )
        
        messages = [
            {
                "role": "user",
                "content": "Generate a math problem that required logic to solve.",
            },
            {"role": "assistant", "content": math_problem},
            {
                "role": "user",
                "content": prompt,
            },
        ]
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=256,
            temperature=0.5,
        )
        response = response.choices[0].message.content
        bt.logging.debug(f"Generated revised math question: {response}")
        return response


if __name__ == "__main__":
    try:
        log_file = open(f'score_logs.txt', 'a')
    except:
        log_file = open(f'score_logs.txt', 'w')
    correctness_cnt = 0
    incorrectness_cnt = 0
    for i in range(10):
        synapse = LogicSynapse(timeout=64)
        challenger = LogicChallenger(base_url, client_key, client_model)
        synapse = challenger(synapse)
        # base_synapse = synapse
        start_time = time.time()
        print(f"*******************************************challege generated!****************************************\n")
    # for i in range(30):
        # synapse = base_synapse
        synapse = solve(synapse)
        end_time = time.time()
        print(f"Time elapsed to solve a problem: {end_time-start_time} seconds")
        synapse.process_time = end_time - start_time

        start_time = time.time()
        rewarder = LogicRewarder(base_url, client_key, client_model)
        similarity, correctness, reward = rewarder(synapse, synapse)
        
        end_time = time.time()
        print(f"Time elapsed to evaluate: {end_time-start_time} seconds")
        print(f"similarity: {similarity} | correctness: {correctness} | reward: {reward}\n")
        print("-"*200)
        print("\n\n\n")
        if correctness == 1:
            correctness += 1
        else:
            incorrectness_cnt += 1
        if correctness < 2:
            log_file.write(f"=====================================================  {i}  =================================================\n")
            log_file.write(f"similarity: {similarity} | correctness: {correctness} | reward: {reward}\n*******************raw***************\n{synapse.raw_logic_question}\n******************* revised *********************\n{synapse.logic_question}\n********************logic answer*************\n{synapse.logic_answer}\n**************ground truth************\n{synapse.ground_truth_answer}\n")
            log_file.write(f"=============================================================================================================\n\n\n")
    
    log_file.close()
    print(correctness)
    print(incorrectness_cnt)
