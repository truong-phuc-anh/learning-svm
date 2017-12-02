import inspect
import logging

def create_logger(file_name, file_level = logging.DEBUG, console_level = logging.DEBUG):
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.DEBUG) #By default, logs all messages

    if console_level != None:
        ch = logging.StreamHandler() #StreamHandler logs to console
        ch.setLevel(console_level)
        ch_format = logging.Formatter('%(asctime)s - %(message)s')
        ch.setFormatter(ch_format)
        logger.addHandler(ch)

    fh = logging.FileHandler(file_name)
    fh.setLevel(file_level)
    fh_format = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(fh_format)
    logger.addHandler(fh)

    return logger


