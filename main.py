from q1_answer import question_1
from q2_answer import question_2

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Enter the question number to run", type=int)
    args = parser.parse_args()

    if args.question == 1:
        question_1()
    elif args.question == 2:
        question_2()
    else:
        print("Invalid question number. Please enter 1 or 2.")


if __name__ == "__main__":
    main()
