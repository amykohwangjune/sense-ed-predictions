import json
import unittest
from app import app

app.testing = True

class TestApi(unittest.TestCase):
    def test_predict(self):
        """
        This function tests the predict function.
        It tests whether the route post the requests successfully and returns a correct response
        """
        with app.test_client() as client:
            tweet1 = "I hate that it's so east to gain the weight back after fasting... #EDProbs"
            tweet2 = "Christmas goals: 100lb or UNDER! I am 104/5 right now... #Skinny4Xmas #thin #proana #ana #mia #ED #skinny"
            tweet3 = "I really want to binge right now, but I have to be strong. I have to resist. #binge #anorexia #thinspo #motivation"
            data = {"data": [{"text_orig": tweet1}, {"text_orig": tweet2}, {"text_orig": tweet3}]}

            result = client.post('/predict', json=data)

            self.assertEqual(200, result.status_code)
            self.assertEqual(int(float(result.data)), 100)
