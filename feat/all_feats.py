import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feat import get_arguments

FILES = ['main.py', 'bureau.py', 'credit_card_balance.py', 'installments_payment.py', 'pos_cash_balance.py',
         'previous_application.py']

if __name__ == '__main__':
    args = get_arguments('all')
    for file in FILES:
        cmd = ['python', '-u', file]
        if args.force:
            cmd += ['-f']
        os.system(' '.join(cmd))
