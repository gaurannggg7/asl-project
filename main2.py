
from translator_gemma import GemmaTranslator
import sys

def main():
    eng_input = "What is your name?"
    translator = GemmaTranslator()
    gloss = translator.to_asl_gloss(eng_input)
    print("Input:", eng_input)
    print("ASL Gloss:", gloss)

if __name__ == "__main__":
    main()
