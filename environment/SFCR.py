
class SFC:
    def __init__(self):
        self.id = None
        self.start_t = None
        self.end_t = None
        self.processing_time = None
        self.vnet = None
        self.bw = None
        
class SFCR(SFC):
    def __init__(self):
        super().__init__()
        self.priority = 0
        self.src = None
        self.embedded_time = None

    """定义优先级规则"""
    def __lt__(self, other):
        if self.priority == other.priority:
            return self.start_t < other.start_t
        return self.start_t < other.start_t
    
    
    
