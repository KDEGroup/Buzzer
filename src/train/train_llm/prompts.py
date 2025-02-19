from typing import Callable


def build_instruction_prompt_alphca(instruction: str) -> str:
    return '''
### Instruction:
{}
### Response:
'''.format(instruction.strip()).lstrip()


def build_instruction_prompt_blank(instruction: str) -> str:
    return '''{}'''.format(instruction.strip()).lstrip()


def build_instruction_prompt_conv(instruction: str) -> str:
    return '''
Human:
{}
Assistant:
'''.format(instruction.strip()).lstrip()


def build_instruction_prompt(prompt_type) -> Callable:
    if prompt_type == 'alphaca':
        return build_instruction_prompt_alphca
    elif prompt_type == 'blank':
        return build_instruction_prompt_blank
    elif prompt_type == 'conv':
        return build_instruction_prompt_conv
    