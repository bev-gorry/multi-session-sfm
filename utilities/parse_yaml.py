import yaml


def parse_yaml(yaml_file):
    with open (yaml_file, 'r') as f:
        args = yaml.safe_load(f)
    
    dataset = args['dataset']
    subset = args['subset']
    
    log_dir = args['log_dir']
    
    dist_threshold = args['dist_threshold']
    
    return dataset, subset, log_dir, dist_threshold
