def print_config(config):
    a=config["a"] or 10
    b=config["b"] or 20
    print("a=",a,"b=",b)
    
config={
    "a":30
}

print_config(config)
