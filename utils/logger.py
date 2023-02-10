import logging
import logging.config
import configparser
import os

configFile = ('../' if __name__ == '__main__' else './') + 'conf/logging.ini'
config = configparser.ConfigParser()
config.read(configFile)

# Default logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        },
    },
    'loggers': {
        '': {
            'handlers': ['console'],
            'level': 'INFO'
        },
    }
}

if config:
    log_file = config.get('DEFAULT', 'log_file')
    log_dir = os.path.dirname(log_file)
    try:
        os.makedirs(log_dir, exist_ok=True) #Make log file directory, as logger expects it to be present
    except FileExistsError:
        pass
    logging.config.fileConfig(config)
else:
    # Loading the default configuration
    logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger(__name__)

# Main method starts here.
if __name__ == '__main__':
    # Example usage
    logger.info('Example log message')
