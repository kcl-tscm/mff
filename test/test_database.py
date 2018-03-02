from create_MFF_database import carve_confs


if __name__ == '__main__':
    # single element (
    filename = 'data/Fe_vac/vaca_iron500.xyz'

    # 2 elements
    filename = 'data/HNi/h_esa500.xyz'

    cutoff = 4.5
    elementslist = carve_confs(filename, cutoff)
