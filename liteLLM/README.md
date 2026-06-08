
# LiteLLM AI Gateway (proxy)

## install

[doc](https://docs.litellm.ai/docs/proxy/docker_quick_start)

stateless install

```
curl -fsSL https://raw.githubusercontent.com/BerriAI/litellm/main/scripts/install.sh | sh
```

- along with the install
  - Config is saved to `./litellm_config.yaml`
  - Master Key is auto-saved in `litellm_config.yaml`
 
- You need a Postgres database for generating keys, users, teams
  - configure by `general_settings.database_url` in `litellm_config.yaml`
## deploy

```
litellm --config ./litellm_config.yaml
```