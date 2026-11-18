import datetime


def on_config(config):
    # Keep the copyright notice in the docs footer current at build time.
    year = datetime.date.today().year
    config.copyright = f"Copyright &copy; 2020-{year} QuantCo Inc."
    return config
