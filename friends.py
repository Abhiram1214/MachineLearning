
class Friends:
    friends = []
    not_friends = []
    employees = ['ram', 'vijay', 'ju']
    
    def __init__(self):
        self.not_friends = self.employees
        
    def display_employees(self):
        print("----------Employees-------------")
        for count, employee in enumerate(Friends.employees, 1):
            print(count, employee) 
        
    
    def display_friends(self):
        for count, friend in enumerate(Friends.friends):
            print(count + ') ' + friend)
    
    def not_friends(self):
        print("ji")
        for count, nonfriend in enumerate(Friends.not_friends, 1):
            print(count, nonfriend)
        

friend = Friends()
friend.display_employees() 
friend.not_friends
friend.employees
friend.not_friends=friend.employees
friend.not_friends()   
    
    
    
while True:
    enter = input("Enter the person you want to friend" )
