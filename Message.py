class Message:
    def __init__(self, sender: int, sender_type: str, recipient: int, recipient_type: str, task_type: str, repeat = 0, content=None):
        self.sender = sender
        self.sender_type = sender_type
        self.recipient = recipient
        self.recipient_type = recipient_type
        self.task_type = task_type
        self.repeat = repeat
        self.content = content


    def reply(self, task_type:str,content = None):
        return Message(self.recipient, self.recipient_type, self.sender, self.sender_type, task_type, content)
    
    def __repr__(self):
        return (f"Message(from={self.sender}, "
                f"to={self.recipient}, "
                f"task={self.task_type}, "
                f"content={self.content})")
