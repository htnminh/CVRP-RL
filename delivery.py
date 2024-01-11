from pprint import pprint, pformat
import numpy as np
from numpy.random import randint
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

class Delivery:
    def __init__(self, n_stops=10, const_vehicle_cap=True, euclidean=True,
                 max_demand=10, max_vehicle_cap=30,
                 seed=None, max_env_size=1_000_000) -> None:
        self.n_stops = n_stops
        self.const_vehicle_cap=const_vehicle_cap
        self.euclidean = euclidean
        self.max_demand = max_demand
        self.max_vehicle_cap = max_vehicle_cap
        assert self.max_demand <= self.max_vehicle_cap
        
        if self.euclidean:
            self.ortools_data, self.other_data = self._create_euclidean_data(seed=seed, max_env_size=max_env_size)
        else:
            pass  # TODO

    
    def _generate_demands_and_vehicle_caps(self):
        demands = np.append([0], randint(1, self.max_demand + 1, size=self.n_stops - 1))

        if self.const_vehicle_cap:
            vehicle_caps = np.full(shape=self.n_stops, fill_value=randint(self.max_demand, self.max_vehicle_cap))
        else:
            vehicle_caps = randint(self.max_demand, self.max_vehicle_cap + 1, self.n_stops)

        return demands, vehicle_caps
    
        
    def _create_euclidean_data(self, seed, max_env_size):
        ortools_data = dict()
        other_data = dict()
        
        if seed is not None:
            np.random.seed(seed)
        
        stops_coords = randint(0, max_env_size + 1, size=(self.n_stops, 2))
        demands, vehicle_caps = self._generate_demands_and_vehicle_caps()

        other_data['stops_coords'] = stops_coords

        ortools_data['distance_matrix'] = cdist(stops_coords, stops_coords)
        ortools_data['demands'] = demands
        ortools_data['vehicle_caps'] = vehicle_caps
        ortools_data['num_vehicles'] = len(vehicle_caps)
        ortools_data['depot'] = 0

        return ortools_data, other_data
        

    def visualize(self, show=True, save_path=None):
        if not self.euclidean:
            raise NotImplementedError("Visualization is only implemented for euclidean 2D space data")
        
        else:
            fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))

            ax1.scatter(self.other_data['stops_coords'][0, 0], self.other_data['stops_coords'][0, 1], color='red', s=10)
            ax1.scatter(self.other_data['stops_coords'][1:, 0], self.other_data['stops_coords'][1:, 1], color='black', s=8)
            for i, txt in enumerate(self.ortools_data['demands']):
                ax1.annotate(txt, (self.other_data['stops_coords'][i, 0], self.other_data['stops_coords'][i, 1]), fontsize=8, textcoords="offset points", xytext=(0, 2), ha='center')
            min_size = np.min(self.other_data['stops_coords'])
            max_size = np.max(self.other_data['stops_coords'])
            ax1.set_xlim([min_size, max_size])
            ax1.set_ylim([min_size, max_size])

            if show:
                plt.show()
            if save_path is not None:
                fig.savefig(save_path, dpi=300)

        return fig, ax1


    def _solve_ortools(self, time_limit_seconds=10):
        # Instantiate the data problem.
        print(f'Solving with time limit of {time_limit_seconds} seconds')

        data = self.ortools_data
        data['distance_matrix'] = data['distance_matrix'].astype('int')
        data['demands'] = data['demands']
        data['vehicle_caps'] = data['vehicle_caps']

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Capacity constraint.
        def demand_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return data["demands"][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data["vehicle_caps"],  # vehicle maximum capacities
            True,  # start cumul to zero
            "Capacity",
        )

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(time_limit_seconds)

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        return data, manager, routing, solution


    def get_ortools_solution(self, time_limit_seconds=10):
        data, manager, routing, solution = self._solve_ortools(time_limit_seconds=time_limit_seconds)
        
        total_distance = 0
        total_load = 0  
        
        routes = list()

        for vehicle_id in range(data["num_vehicles"]):
            route = dict()
            
            route_distance = 0
            route_load = 0
            route_segments = list()

            index = routing.Start(vehicle_id)            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)

                route_load += data["demands"][node_index]                
                route_segments.append(dict(index=node_index, load=route_load))

                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )

            route_segments.append(dict(index=manager.IndexToNode(index), load=route_load))

            route['segments'] = route_segments
            route['distance'] = route_distance
            routes.append(route)

            total_distance += route_distance
            total_load += route_load

        return dict(
            objective=solution.ObjectiveValue(),
            total_distance=total_distance,
            total_load=total_load,
            routes=routes
        )


    def visualize_ortools_solution(self, show=True, time_limit_seconds=10):
        _color_list = list(mcolors.TABLEAU_COLORS.keys())
        fig, ax1 = self.visualize(show=False)
        solution = self.get_ortools_solution(time_limit_seconds=time_limit_seconds)
        routes = filter(lambda route: route['distance'] > 0, solution['routes'])

        for _i, route in enumerate(routes):
            _color = _color_list[_i % len(_color_list)]
            route_segments = route['segments']
            route_distance = route['distance']
            route_load = route_segments[-1]['load']
            route_coords = self.other_data['stops_coords'][[x['index'] for x in route_segments], :]

            ax1.plot(route_coords[:, 0], route_coords[:, 1], linewidth=1, color=_color)
            ax1.annotate(f'{route_load}',
                         ((route_coords[-1, 0] + route_coords[-2, 0])/2, (route_coords[-1, 1] + route_coords[-2, 1])/2),
                         fontsize=8, textcoords="offset points", xytext=(0, 0), ha='center', color=_color)

        if show:
            plt.show()

if __name__ == "__main__":
    delivery = Delivery(n_stops=20, const_vehicle_cap=True)
    pprint(delivery.other_data)
    pprint(delivery.ortools_data)
    delivery.visualize()
    # pprint(delivery.get_ortools_solution(time_limit_seconds=1))
    pprint(delivery.visualize_ortools_solution(time_limit_seconds=60))
