
class Loader:
    loader_counter = 0
    loaders = []

    def __init__(self, loading_rate, mass_independent_loading_time, loader_id):
        self.loading_rate = loading_rate    # s/ tonn
        self.loader_id = loader_id
        self.mass_independent_loading_time = mass_independent_loading_time

    def set_location_node(self, location_node):
        self.location_node = location_node

    def get_loading_rate(self):
        """
        Will depend on the loading_rate

        """

        return self.loading_rate
    
    def get_mass_independent_loading_time(self):
        return self.mass_independent_loading_time
