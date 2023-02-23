import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(funcName)s() - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
