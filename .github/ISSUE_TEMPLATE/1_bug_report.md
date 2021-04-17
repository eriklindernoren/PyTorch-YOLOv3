---
name: "\U0001F41B Bug report"
about: Report a bug, crash or some misbehavior
title: ''
labels: 'bug'
assignees: ''
---
<!--- Provide a general summary of the issue in the title above -->

## Context
<!--- Please provide context, as this streamlines the debugging process. Mark the correct cases and follow the instructions. -->
- [ ] I have installed this repo manually and the issue occurred on this commit:
<!--- Get the current commit hash either from the first printout of the program or by executing the following command: 'git rev-parse --short HEAD' -->
- [ ] I have installed this repo via `PIP` and the issue occurred on version: <!--- Get the current version number by executing the following command: 'pip show pytorchyolo' -->
- [ ] The issue occurred when using the following .cfg model:
    - [ ] `yolov3`
    - [ ] `yolov3-tiny`
    - [ ] `CUSTOM`

## Necessary Checks
<!--- Please ensure, you have completed the following checks. This helps to give insight into the issue and prevent already resolved issues. -->
- [ ] The issue occurred on the newest version
<!--- If installed manually, run: 'git pull && poetry install'  -->
<!--- If installed via PIP, run: 'pip install --upgrade pytorchyolo' -->
- [ ] I couldn't find a similar issue here on this project's github repo
- [ ] If the issue is CUDA related (CUDA error), I have tested and provided the traceback also when CUDA is turned off <!--- For linux, rerun your steps with the prefix CUDA_VISIBLE_DEVICES="" -->
- [ ] I have provided all tracebacks or printouts in ```Text Form``` <!--- This makes it easier to search for errors. -->
- [ ] In case, the issue occurred on a custom .cfg model, I have provided the model down below

## Expected behavior
<!--- Describe what you expected to happen -->

## Current behavior
<!--- Describe what actually happened instead of the expected behavior -->

## Steps to Reproduce
<!--- An unambiguous set of steps to reproduce this bug. -->
<!--- Code-snippets, screenshots ot other details are welcome if needed. -->
1.
2.
3.
...

## Possible Solution
<!--- If you already have an idea, you can suggest a fix/reason for the bug. This is not obligatory. -->

<!--- Please remove the following block, if this does not apply to you issue. -->
### Custom `.cfg`
<!--- Please paste your custom .cfg model below. -->
<details><summary>Custom .cfg</summary>
<p>
<!--- YOUR CUSTOM .CFG HERE -->
</p>
</details>
