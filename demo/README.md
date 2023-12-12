## BSChecker Demo

You can run the BSChecker on your server. By default, it applies GPT-4 as the extractor and the NLI model as the checker. 

### Run the Demo Locally

To run the demo, first set the OpenAI API key for the extractor, and Serper API key to use Google seach:
```bash
export OPENAI_API_KEY=<your openai api key>
export SERPER_API_KEY=<your serper api key>
```

Then run the following command:

```bash
streamlit run demo/main.py
```

It will print the URL for of the demo, you can open the URL in your browser to interact with the demo.
