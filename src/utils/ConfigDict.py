class ConfigDict(dict):
    """
    自定义Config字典
    """

    def __getitem__(self, key):
        if key not in self and '.' in key:
            ks = key.split('.')
            v = self[ks[0]]
            for k in ks[1:]:
                v = v[k]
            return v
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if key not in self and '.' in key:
            ks = key.split('.')
            v = self[ks[0]]
            for k in ks[1:-1]:
                if k not in v:
                    v[k] = ConfigDict()
                v = v[k]
            v[ks[-1]] = value
        super().__setitem__(key, value)

    def get(self, key, default=None):
        if key not in self and '.' in key:
            ks = key.split('.')
            v = super().get(ks[0], default)
            for k in ks[1:]:
                if v is default or not hasattr(v, 'get'):
                    return default
                v = v.get(k, default)
            return v
        return super().get(key, default)
