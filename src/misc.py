import yaml

def load_prompt_template(path_to_template_txt):
    f = open(path_to_template_txt, "r") 
    prompt_template = f.read()
    f.close()
    return prompt_template

def load_config(config_path):
    f = open(config_path)
    config = yaml.load(f, Loader=yaml.FullLoader)
    return config





