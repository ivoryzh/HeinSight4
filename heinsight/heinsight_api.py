import requests

session = requests.Session()


class HeinsightAPI:
    def __init__(self, address, source=None, fps=10):
        self.address = address
        self.time_course_data = []
        self.stream_url = f"{address}/frame"
        self.running = False
        self.source = source
        self.fps = fps

    def start(self):
        self.running = True
        return session.post(self.address + '/start', json={"video_source": self.source, "frame_rate": self.fps}).json()

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

    def _get_status(self, hs_class):
        response = session.get(self.address + '/current_status')
        if response.status_code >= 400:
            return None
        status = response.json().get("status")
        return status.get(hs_class, False)


if __name__ == "__main__":
    heinsight = HeinsightAPI("http://localhost:8080")
    print(heinsight)