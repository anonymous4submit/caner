from data import ORG_A_CSV


def get_domain_list():
    # Define the list of domains
    domain_list = ['Utilities', 'RealEstate', 'Consumer', 'Industry', 'Materials',
                   'Financial', 'InformationTechnology', 'MedicalHealth', 'Telecom']
    return domain_list


def get_org_dict():
    # Return the org's domain dictionary, and org list
    org_dict = {}
    with open(ORG_A_CSV, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            key = line.split(',')[0].strip()
            value = line.split(',')[1].strip()
            org_dict[key] = value
    return org_dict


def get_domain_target(source_list, target_list):
    """
    According to the original source list and target list, return the domain target list
    :param source_list:
    :param target_list:
    :return: domain target list
    """
    org_dict = get_org_dict()
    domain_target_list = []

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
        domain_target_list.append(new_t_line)
    return domain_target_list


def get_domain_dict_by_csv(csv_list):
    """
    Given a list of domain csv, return a dict of 'mention -> domain label'
    """
    domain_dict = {}
    for csv_f in csv_list:
        with open(csv_f, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                key = line.split(',')[0].strip()
                value = line.split(',')[1].strip()
                domain_dict[key] = value
    return domain_dict


def get_domain_target_by_dict(domain_dict, source_list, target_list):
    """
    According to the original source list and target list, return the domain target list
    """
    domain_target_list = []

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
                if cur_org in domain_dict:
                    org_type = domain_dict[cur_org]
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
        domain_target_list.append(new_t_line)
    return domain_target_list
