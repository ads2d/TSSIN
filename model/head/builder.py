import model


def build_head(cfg):
    param = dict()
    for key in cfg:
        if key == 'type':
            continue
        param[key] = cfg[key]

    head = model.head.__dict__[cfg['type']](**param)

    return head
