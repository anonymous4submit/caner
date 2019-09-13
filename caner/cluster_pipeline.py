from data import ORG_A_CSV


def get_cluster_list():
    # Define the list of clusters
    cluster_list = ['Utilities', 'RealEstate', 'Consumer', 'Industry', 'Materials',
                   'Financial', 'InformationTechnology', 'MedicalHealth', 'Telecom']
    return cluster_list


def get_org_dict():
    # Return the org's cluster dictionary, and org list
    org_dict = {}
    with open(ORG_A_CSV, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            key = line.split(',')[0].strip()
            value = line.split(',')[1].strip()
            org_dict[key] = value
    return org_dict


def get_cluster_target(source_list, target_list):
    """
    According to the original source list and target list, return the cluster target list
    :param source_list:
    :param target_list:
    :return: cluster target list
    """
    org_dict = get_org_dict()
    cluster_target_list = []

    for s_line, t_line in zip(source_list, target_list):
        assert len(s_line) == len(t_line)
        new_t_line = []
        left = right = -1
        for i in range(len(s_line)):
            if left <= i < right:
                continue
            if t_line[i][0] == 'B':
                left = i
                right = left + 1
                while right < len(s_line):
                    if t_line[right][0] != 'I':
                        break
                    right += 1
                cur_org = ''.join(s_line[left:right])
                if cur_org in org_dict:
                    org_type = '-'+org_dict[cur_org]
                else:
                    # print(cur_org)
                    org_type = ''
                for j in range(left, right):
                    if org_type != '':
                        new_t_line.append((t_line[j]+org_type)[2:])
                    else:
                        new_t_line.append((t_line[j] + org_type))
            else:
                new_t_line.append(t_line[i])
        cluster_target_list.append(new_t_line)
    return cluster_target_list


def get_cluster_dict_by_csv(csv_list):
    """
    Given a list of cluster csv, return a dict of 'mention -> cluster label'
    """
    cluster_dict = {}
    for csv_f in csv_list:
        with open(csv_f, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                key = line.split(',')[0].strip()
                value = line.split(',')[1].strip()
                cluster_dict[key] = value
    return cluster_dict


def get_cluster_target_by_dict(cluster_dict, source_list, target_list):
    """
    According to the original source list and target list, return the cluster target list
    """
    cluster_target_list = []

    for s_line, t_line in zip(source_list, target_list):
        assert len(s_line) == len(t_line)
        new_t_line = []
        left = right = -1
        for i in range(len(s_line)):
            if left <= i < right:
                continue
            if t_line[i][0] == 'B':
                left = i
                right = left + 1
                while right < len(s_line):
                    if t_line[right][0] != 'I':
                        break
                    right += 1
                cur_org = ''.join(s_line[left:right])
                if cur_org in cluster_dict:
                    org_type = cluster_dict[cur_org]
                else:
                    # print(cur_org)
                    org_type = ''
                for j in range(left, right):
                    if org_type != '':
                        new_t_line.append(org_type)
                    else:
                        new_t_line.append((t_line[j] + org_type))
            else:
                new_t_line.append(t_line[i])
        cluster_target_list.append(new_t_line)
    return cluster_target_list
