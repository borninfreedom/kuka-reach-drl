# Python Knowledge
* [Python中的作用域、global与nonlocal](https://note.qidong.name/2017/07/python-legb/)
* [Delgan/loguru](https://github.com/Delgan/loguru), this is a great python log module, it is much greater than python built in logging module.
* [wandb](https://wandb.ai/site),Developer tools for machine learning. Build better models faster with experiment tracking, dataset .
* logging usage
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    filename='./logs/client1-{}.log'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())),
    filemode='w')
logger = logging.getLogger(__name__)

formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
stream_handler = logging.StreamHandler()

stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# in the codes.
# logger.info()
# logger.debug()
```
