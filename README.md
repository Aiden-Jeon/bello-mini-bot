# bello-mini-bot

## Installation

### Poetry

Install packages with poetry.

```python
poetry install
```

### Litellm Proxy

1. copy `config_template.yaml` file and write the desired keys

    ```bash
    cp docker/litellm/config_template.yaml docker/litellm/config.yaml
    ```

    Support keys are below:

    - huggignface : [Link](https://huggingface.co/settings/tokens)
    - openai: [Link](https://platform.openai.com/settings/organization/api-keys)

2. run litellm server with docker compose.

    ```bash
    docker compose up -d
    ```

3. check the litellm server. If key are not provided it will not work.
    ```bash
    make test-litellm
    ```

### Usage

Using litellm's api base server `http://0.0.0.0:4000`, openai_api_key must be `sk-1234`.
This is key that is written on `docker/litellm/config.yaml`.

```python
ChatOpenAI(
        openai_api_base="http://0.0.0.0:4000",
        model="openai/gpt-3.5-turbo",
        openai_api_key="sk-1234",
    )
```
