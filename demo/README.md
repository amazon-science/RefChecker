## RefChecker Demo

You can run the RefChecker on your server. 

### Export API Keys

To run the demo, we should first set the relevant API keys for the extractor and checker.

- If we use OpenAI models (i.e. GPT-4), run the following command:
```bash
export OPENAI_API_KEY=<your openai api key>
```

- To use Claude 2, if we have an Anthropic API Key, run:
```bash
export ANTHROPIC_API_KEY=<your anthropic api key>
```

If we are using Claude 2 on AWS Bedrock and running the demo on AWS, we need to export the region of the server:

```bash
export aws_bedrock_region=<your aws bedrock region, e.g. us-west-2>
```

- If we want to use Google search to find references, export the Serper API key:

```bash
export SERPER_API_KEY=<your serper api key>
```

### Run the Demo

Execute the following command to run the demo:

```bash
streamlit run demo/main.py \
    --server.port={port} -- \ # set the deploy port
    --enable_search # enable Google search
```

It will print the URL of the demo, you can open it in your browser to interact with the demo.
