import sys

def error_message_detail(error,error_detail:sys): # pyright: ignore[reportGeneralTypeIssues]
    _,_,exc_tb=error_detail.exc_info()

    file_name=exc_tb.tb_frame.f_code.co_filename # type: ignore

    return f"Error occured in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]" # type: ignore
    

class CustomException(Exception):

    def __init__(self,error_message,error_detail:sys): # type: ignore
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail)

    def __str__(self):
        return self.error_message