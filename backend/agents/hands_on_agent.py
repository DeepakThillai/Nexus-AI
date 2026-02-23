from groq import Groq

class HandsOnAgent:
    def __init__(self):
        # Use the working pattern
        self.client = Groq(
            api_key="XXX"
        )

        # This is our "memory"
        self.conversation_history = [
            {
                "role": "system",
                "content": (
                    "You are a hands-on technical mentor.\n"
                    "Assign ONE practical real-world task for the given target role.\n"
                    "Guide the user step-by-step.\n"
                    "Wait for confirmation before moving forward.\n"
                    "Clarify doubts clearly.\n"
                    "Only conclude the session when satisfied or user types '$'.\n"
                    "Structure responses as:\n\n"
                    "Task:\n"
                    "Current Step:\n"
                    "What To Do:\n"
                    "Reply After Completion:\n"
                )
            }
        ]

    def ask_gpt(self):
        completion = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=self.conversation_history,
            max_completion_tokens=1500,
            stream=False
        )

        return completion.choices[0].message.content

    def start_session(self):
        target_role = input("Enter your target role: ")

        # Add user request
        self.conversation_history.append({
            "role": "user",
            "content": f"My target role is {target_role}. Assign my first hands-on task."
        })

        response = self.ask_gpt()
        print("\n" + response)

        # Store assistant reply
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        self.interactive_loop()

    def interactive_loop(self):
        while True:
            user_input = input("\nYou: ")

            if user_input.strip() == "$":
                print("Session ended by user.")
                break

            # Add user input to memory
            self.conversation_history.append({
                "role": "user",
                "content": user_input
            })

            response = self.ask_gpt()
            print("\n" + response)

            # Store assistant reply
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })

            # Optional auto-stop if GPT concludes
            if "task completed" in response.lower():
                print("\nAgent has marked task as completed.")
                break


if __name__ == "__main__":
    agent = HandsOnAgent()
    agent.start_session()
