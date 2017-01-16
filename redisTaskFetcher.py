import redis
import ast

KEY_IMG_URL = "img"
KEY_ID = "id"

"""http://peiqiang.net/2014/12/31/python-simple-queue-redis-queue.html"""
"""http://blog.fens.me/linux-redis-install/"""

class RedisQueue(object):
    """Simple Queue with Redis Backend"""
    def __init__(self, name, namespace='queue', **redis_kwargs):
        """The default connection parameters are: host='localhost', port=6379, db=0"""
        self.__db= redis.Redis(**redis_kwargs)
        self.key = '%s:%s' %(namespace, name)

    def qsize(self):
        """Return the approximate size of the queue."""
        return self.__db.llen(self.key)

    def empty(self):
        """Return True if the queue is empty, False otherwise."""
        return self.qsize() == 0

    def put(self, item):
        """Put item into the queue."""
        self.__db.rpush(self.key, item)

    def get(self, block=True, timeout=None):
        """Remove and return an item from the queue.

        If optional args block is true and timeout is None (the default), block
        if necessary until an item is available."""
        if block:
            item = self.__db.blpop(self.key, timeout=timeout)
        else:
            item = self.__db.lpop(self.key)

        if item:
            item = item[1]
        return item

    def get_nowait(self):
        """Equivalent to get(False)."""
        return self.get(False)

REDIS_QUEUE = RedisQueue("test", namespace='test', host='localhost', port=6379, db=0)

def testResdisServer():
    if REDIS_QUEUE.empty():
        REDIS_QUEUE.put("{'id':111, 'img':'http://aaa.aaa.aaa/aaa.jpg'}")

    print "qsize:", REDIS_QUEUE.qsize()
    #print "empty:", REDIS_QUEUE.empty()
    ret = getTaskFromQueue()
    print type(ret), ret

def getTaskFromQueue():
    # Should return something like {'id':111, 'img':'http://aaa.aaa.aaa/aaa.jpg'}
    ret = REDIS_QUEUE.get(True)
    return ast.literal_eval(ret)

if __name__ == '__main__':
    testResdisServer()
