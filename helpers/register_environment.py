import root
from tools.sumo_tools import generate_random_flow
from tools.route_validator import validate_routes
from tools.calculate_volume import VolumeData
from tools.call_route_sampler import RouteSampler
from tools.pull_sql_data import Connection
from tools.traffic_lights.traffic_light_manager import TrafficLightManagerFSM


AGG_PERIOD = 600


def generate_input_files(input_dict):

    random_output = ".".join(
                             [
                                 input_dict['route_file'][:input_dict['route_file'].find('.xml')],
                                 'random',
                                 'xml'
                              ]
                         )
    validated_random_output = ".".join(
                                         [
                                             input_dict['route_file'][:input_dict['route_file'].find('.xml')],
                                             'random_validated',
                                             'xml'
                                          ]
                                        )

    generate_random_flow(net_file_path=input_dict['net_file'],
                         validate=True,
                         period=1,
                         output_file_path=random_output,
                         sim_length=3600,
                         fringe_factor=str(1000),
                         lane_factor=str(3),
                         min_distance=str(30),
                         vehicle_class='vehDist',
                         output_validated_path=validated_random_output
                         )

    # remove routes that do not end at an edge
    validate_routes(
                    net_path=input_dict['net_file'],
                    route_path=validated_random_output,
                    save_path=validated_random_output
                    )

    conn = Connection(root.SQL_CONNECTION_FILE, time_offset={'63082004': 6})
    conn.pull_data(['63082002', '63082003', '63082004'], input_dict['start_datetime'], input_dict['end_datetime'])
    raw_data = conn.get_data()

    vd = VolumeData(raw_data, start_time=input_dict['start_datetime'], end_time=input_dict['end_datetime'], )
    volume_obj, time_index = vd.calculate_volume(AGG_PERIOD, hourly=False, correction_file=root.CORRECTION_FILE)

    rs = RouteSampler(net_file=input_dict['net_file'],
                      connection_file=root.DETECTOR_LOCATION_FILE,
                      volume=volume_obj,
                      time_index=time_index,
                      skip_detectors={'63082002': [2, 6], '63082003': [2, 6], '63082004': [2, 6]},
                      turn_ratio_file_path=root.TURN_RATIOS
                      )

    rs.generate_input_xml(turn_file_path=root.ROUTESAMPLER_INPUT)

    rs.run_route_sampler(random_route_file=root.VALIDATED_RANDOM_ROUTE,
                         output_file_path=root.ROUTESAMPLER_ROUTE_FILE,
                         depart_speed='max',
                         vehicle_class='vehDist',
                         turn_file_path=root.ROUTESAMPLER_INPUT)

    TrafficLightManagerFSM(end_time=SIM_LENGTH,
                           sim_step=SIM_STEP,
                           tl_settings_file=root.INTERSECTION_SETUP,
                           rw_start_time=START_TIME,
                           historical_tl_data=vd.get_light_events(),
                           tls_file=root.TLS_FILE
                           )
