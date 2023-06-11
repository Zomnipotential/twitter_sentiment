import packages as pkg
import constants as const

def title(message):
	print(f'{3*const.nl}{(len(message)+4)*"*"}{const.nl}* {message} *{const.nl}{(len(message)+4)*"*"}{2*const.nl}')