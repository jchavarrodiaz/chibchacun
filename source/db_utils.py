from configparser import ConfigParser


def get_config_params(filename='../config/database.ini', section='prestodb_cassandra'):
    """
    Returns dictionary with configuration for accessing data bases.
    :param filename: .ini file name
    :param section: section .ini file for reading.
    :return:
    """
    parser = ConfigParser()
    parser.read(filename)

    db = {}

    if parser.has_section(section):
        params = parser.items(section)

        for param in params:
            db[param[0]] = param[1]

    else:

        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db


if __name__ == '__main__':
    pass
