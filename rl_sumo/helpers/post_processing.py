"""
This file reads the emissions output file of SUMO. Runs a little faster than SUMO's xml2csv.py script, 
but not as fast as possible with reading file into memory 
"""
import csv
from lxml import etree

FIELD_NAMES = {
    'emissions': [
        'vehicle_CO', 'vehicle_CO2', 'vehicle_HC', 'vehicle_NOx', 'vehicle_PMx', 'vehicle_angle', 'vehicle_eclass',
        'vehicle_electricity', 'vehicle_fuel', 'vehicle_id', 'vehicle_lane', 'vehicle_noise', 'vehicle_pos',
        'vehicle_route', 'vehicle_speed', 'vehicle_type', 'vehicle_waiting', 'vehicle_x', 'vehicle_y'
    ],
    'e1': [
        'interval_begin', 'interval_end', 'interval_flow', 'interval_harmonicMeanSpeed', 'interval_id',
        'interval_length', 'interval_nVehContrib', 'interval_nVehEntered', 'interval_occupancy'
    ],
    'e2': [
        'interval_begin',
        'interval_end',
        'interval_id',
        'interval_sampledSeconds',
        'interval_nVehEntered',
        'interval_nVehLeft',
        'interval_nVehSeen',
        'interval_meanSpeed',
    ]
}


def _parse_and_write_emissions(elem, csv_writer, fields, metadata):
    if (elem.tag == 'timestep') and (len(elem.attrib) > 0):
        metadata = [elem.attrib['time']]
        return metadata
    elif (elem.tag == 'vehicle') and (len(elem.attrib) >= 19):
        csv_writer.writerow(metadata + [elem.attrib[col_name] for col_name in fields])
        return metadata
    return metadata


def _parse_and_write_detector(elem, csv_writer, fields, metadata):
    try:
        csv_writer.writerow([elem.attrib[key] for key in fields])
    except KeyError:
        return 0


PARSE_FUNCTION = {
    'emissions': _parse_and_write_emissions,
    'e1': _parse_and_write_detector,
    'e2': _parse_and_write_detector
}


class _XML2CSV:
    def __init__(self, file_path, fields, xml_fields, parse_function, save_path=None):
        self._csv_writer = None
        self._file_path = file_path
        self._fields = fields
        self._fields_simp = xml_fields
        self._save_path = save_path
        self._parse_function = parse_function
        self.main()

    @staticmethod
    def fast_iter(context, func, **kwargs):
        meta_data = [0]
        try:
            for _, elem in context:
                meta_data = func(elem, metadata=meta_data, **kwargs)
                elem.clear()
                while elem.getprevious() is not None:
                    try:
                        del elem.getparent()[0]
                    except TypeError:
                        break
            del context
        except Exception as e:
            print(e)

    def main(self, ):
        header_fields = self._fields
        fields = self._fields_simp
        context = etree.iterparse(self._file_path, events=("start", "end"))
        with open(self._save_path, mode='w+') as file:
            self._csv_writer = csv.writer(file, delimiter=',')
            self._csv_writer.writerow(header_fields)
            self.fast_iter(context, func=self._parse_function, csv_writer=self._csv_writer, fields=fields)


def xml2csv(file_path: str, file_type: str, save_path: str):
    xml_fields = [col_name.split(sep="_")[-1] for col_name in FIELD_NAMES[file_type]]
    header_fields = ['interval_begin'] + FIELD_NAMES[file_type] if file_type in 'emissions' else FIELD_NAMES[file_type]
    _XML2CSV(file_path=file_path,
             fields=header_fields,
             xml_fields=xml_fields,
             parse_function=PARSE_FUNCTION[file_type.lower()],
             save_path=save_path)