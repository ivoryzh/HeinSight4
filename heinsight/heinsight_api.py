import requests

session = requests.Session()


class HeinsightAPI:
    def __init__(self, address, source=None, fps=10, res=(1920, 1080)):
        self.address = address
        self.time_course_data = []
        self.stream_url = f"{address}/frame"
        self.running = False
        self.source = source
        self.fps = fps
        self.res = res

    def start(self):
        self.running = True
        return session.post(self.address + '/start',
                            json={
                                "video_source": self.source,
                                "frame_rate": self.fps,
                                "res": self.res,
                            }).json()

    def stop(self):
        self.running = False
        return session.get(self.address + '/stop').json()

    def data(self):
        data = session.get(self.address + '/data').json()
        return data

    def get_current_status(self):
        data = session.get(self.address + '/current_status').json()
        return data

    def homo(self):
        return self._get_status("Homo")

    def hetero(self):
        return self._get_status("Hetero")

    def empty(self):
        return self._get_status("Empty")

    def residue(self):
        return self._get_status("Residue")

    def solid(self):
        return self._get_status("Solid")

    def turbidity(self, rolling_average:int=1):
        return self._get_data("turbidity", rolling_average)

    def turbidity_1(self, rolling_average:int=1):
        return self._get_data("turbidity_1", rolling_average)

    def turbidity_2(self, rolling_average:int=1):
        return self._get_data("turbidity_2", rolling_average)

    def volume_1(self, rolling_average:int=1):
        return self._get_data("volume_1", rolling_average)

    def volume_2(self, rolling_average:int=1):
        return self._get_data("volume_2", rolling_average)

    def _get_data(self, data_class, rolling_average):
        if rolling_average == 1 or rolling_average == 0 or rolling_average is None or rolling_average is False:
            response = session.get(self.address + '/current_status')
            if response.status_code >= 400:
                return None
            data = response.json().get("data")
            return data.get(data_class, None)
        else:
            response = session.get(self.address + '/rolling_data')
            if response.status_code >= 400:
                return None #, 0
            data = response.json().get("hsdata")
            last_data = data[-rolling_average:] if len(data) > rolling_average else data
            data_queue = []
            for i in last_data:
                data_queue.append(i.get(data_class, False))
            data_queue = [value for value in data_queue if value]
            if len(data_queue) == 0:
                return None #, 0
            else:
                return sum(data_queue) / len(data_queue) #, len(data_queue)/rolling_average


    def _get_status(self, hs_class):
        response = session.get(self.address + '/current_status')
        if response.status_code >= 400:
            return None
        status = response.json().get("status")
        return status.get(hs_class, False)


if __name__ == "__main__":
    heinsight = HeinsightAPI("http://localhost:8000", source=0, res=(1920, 1080))
    print(heinsight)