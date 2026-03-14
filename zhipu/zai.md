# Z.ai
[chat](https://chat.z.ai/)
[API](https://z.ai/model-api)

## Troubleshoot
Q: Why does it still report error "1113 Insufficient Balance" after purchasing the coding package? Why is the account balance still deducted after purchasing the coding package?
A: The situation of reporting insufficient balance or deducting account balance may be due to not meeting the usage conditions of the GLM Coding Plan coding package:
1. The coding plan currently supports Claude Code、Open Code、Cline、Factory Droid、Kilo Code、Roo Code、Crush、Goose、Cursor、Gemini CLI、Grok CLI、Cherry studio.
2. A specific baseurl address must be configured to use it:
- API endpoint for Claude Code and Goose is：https://api.z.ai/api/anthropic
- API endpoint for other tools is: https://api.z.ai/api/coding/paas/v4
  - work also `base_url` in SDK  


## devpack 
coding helper
- `npx @z_ai/coding-helper`
- support Claude Code, OpenCode, Crush, Factory Droid

Tool guide
- [claude code](https://docs.z.ai/devpack/tool/claude)
  