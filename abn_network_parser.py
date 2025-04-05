# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 20:42:48 2025

@author: faith
"""





from enum import Enum
import copy


class Precedence(Enum):
    LOWEST = 0
    STATEMENT = 1
    ASSIGN = 2
    NOT_EQUAL = 3,
    AND_OR = 4
    LESS_THAN = 5
    GREATER_THAN = 6
    EQUALS = 7
    SUM = 8
    PRODUCT = 9
    PREFIX = 10
    CALL = 11
class TokenKind(Enum):
     EOF = 1 
     ENDLINE = 2
     L_PAREN = 3
     R_PAREN = 4
     PLUS = 5
     MINUS = 6
     F_SLASH = 7
     B_SLASH = 8
     L_CURLY = 9
     R_CURLY = 10
     EQUALITY = 11
     LESS_THAN = 12
     GREATER_THAN = 13
     LESS_OR_EQUAL = 14
     GREATER_OR_EQUAL = 15
     AND = 16
     OR = 17
     NOT = 18
     AMPERSAND = 19
     N_LITERAL = 20
     S_LITERAL = 21
     STAR = 22
     COLON = 23
     COMMA = 24
     L_SQUARE = 25
     R_SQUARE = 26
     PIPE = 27
     MODULO = 28
     DOT = 29
     NOT_EQUAL = 30
     LOGICAL_OR = 31
     LOGICAL_AND = 32
     LOGICAL_NOT = 33
     XOR = 34
     IDENTIFIER = 35
     ASSIGN = 36
     
     
class Token:
     def __init__(self, kind, literal):
         self.kind = kind
         self.literal = literal
class Lexer:
     def __init__(self, source):
         self.source = source.strip()
         self.tokens = []
         self.position = -2
         self.c_char = '\0'
         self.n_char = '\0'
         self.p_char = '\0'
         
         self.next()
         
     def is_whitespace_char(self, char):
         return char in '\t '
     def is_newline_char(self, char):
         return char in '\r\n'
     def consume_whitespace(self):
         while True:
             if self.is_newline_char(self.c_char):
                 break
             if self.c_char != '\0' and self.is_whitespace_char(self.c_char):
                 self.next()
             else:
                 break
                           
         
     def lex(self):
         token = Token
         self.next()
         #are we on EOF?
         self.consume_whitespace()
         c_char = self.c_char
         char_str = f'{c_char}'
         if c_char == ';':
             return Token(TokenKind.ENDLINE, ';')
         elif self.is_newline_char(self.c_char):
             return Token(TokenKind.ENDLINE, '\\n')  # Emit an ENDLINE token
         elif c_char == '\0':
             return Token(TokenKind.EOF, '\0')
         elif c_char == '(':
             return Token(TokenKind.L_PAREN, '(')
         elif c_char == ')':
             return Token(TokenKind.R_PAREN, ')')
         
         elif self.is_letter(c_char):
             token = self.make_identifier()
             if self.is_reserved_word(token.literal):
                 return self.make_reserved_word_token(token)
             #else it's probably just an identifier
             return token
         elif c_char == '=':
             n_char = c_char
             if self.n_char == '=':
                 n_char = f'{n_char}{self.n_char}'
                 return Token(TokenKind.EQUALITY, n_char)
             else:
                 return Token(TokenKind.ASSIGN, n_char)
         elif c_char == '<':
             n_char = c_char
             if self.n_char == '=':
                 n_char = f'{n_char}{self.n_char}'
                 return Token(TokenKind.LESS_OR_EQUAL, n_char)
             else:
                 return Token(TokenKind.LESS_THAN, n_char)
             
         elif c_char == '>':
             n_char = c_char
             if self.n_char == '=':
                 n_char = f'{n_char}{self.n_char}'
                 self.next()
                 return Token(TokenKind.GREATER_OR_EQUAL, n_char)
             else:
                 return Token(TokenKind.GREATER_THAN, n_char)
         elif c_char == '&':
             n_char = c_char
             if self.n_char == '&':
                 n_char = f'{n_char}{self.n_char}'
                 self.next()
                 return Token(TokenKind.AND, n_char)
             else:
                 return Token(TokenKind.AMPERSAND, n_char)
             
         elif c_char == '\"' or c_char == '\'':
             #beginning of a string literal e.g "Faith"
             return self.make_string_literal()
         elif self.is_digit(c_char):
             return self.make_numeric_literal()
         elif c_char == '+':
             return Token(TokenKind.PLUS, c_char)
         elif c_char == '-':
             return Token(TokenKind.MINUS, c_char)
         elif c_char == '*':
             return Token(TokenKind.STAR, c_char)
         elif c_char == '/':
             return Token(TokenKind.F_SLASH, c_char)
         elif c_char == ':':
             return Token(TokenKind.COLON, c_char)
         elif c_char == ',':
             return Token(TokenKind.COMMA, c_char)
         elif c_char == '{':
             return Token(TokenKind.L_CURLY, c_char)
         elif c_char == '}':
             return Token(TokenKind.R_CURLY, c_char)
         elif c_char == '[':
             return Token(TokenKind.L_SQUARE, c_char)
         elif c_char == ']':
             return Token(TokenKind.R_SQUARE, c_char)
         elif c_char == '!':
             n_char = c_char
             if self.n_char == '=':
                 n_char = f'{n_char}{self.n_char}'
                 return Token(TokenKind.NOT_EQUAL, n_char)
             else:
                 return Token(TokenKind.NOT, n_char)
         elif ord(c_char) == 0x2227: #logical OR
             #{0x2227, 0x2228}
             return Token(TokenKind.LOGICAL_OR, c_char)
         elif ord(c_char) == 0x2228: #logical AND
             #{0x2227, 0x2228}
             return Token(TokenKind.LOGICAL_AND, c_char)
         elif ord(c_char) == 0x00AC: #logical NOT
             #{0x2227, 0x2228}
             return Token(TokenKind.LOGICAL_NOT, c_char)
         elif ord(c_char) == 0x2295: #XOR
             #{0x2227, 0x2228}
             return Token(TokenKind.XOR, c_char)
         elif c_char == '|':
             n_char = c_char
             if self.n_char == '|':
                 n_char = f'{n_char}{self.n_char}'
                 return Token(TokenKind.OR, n_char)
             else:
                 return Token(TokenKind.PIPE, n_char)
         elif c_char == '%':
             return Token(TokenKind.MODULO, c_char)
         elif c_char == '.':
             return Token(TokenKind.DOT, c_char)
         
         
         return Token(TokenKind.EOF, '\0')
             
     def is_reserved_word(self, literal):
         return literal.lower() in ['and', 'or', 'xor', 'not']
     def make_reserved_word_token(self, token: Token):
         if token.literal.lower() == 'and':
             return Token(TokenKind.LOGICAL_AND, token.literal)
         elif token.literal.lower() == 'or':
             return Token(TokenKind.LOGICAL_OR, token.literal)
         elif token.literal.lower() == 'not':
             return Token(TokenKind.LOGICAL_NOT, token.literal)
         elif token.literal.lower() == 'xor':
             return Token(TokenKind.XOR, token.literal)
         return token
     def make_numeric_literal(self):
         curr_pos = self.position
         dots = 0
         while True:
             if self.c_char != '\0' and self.is_digit(self.n_char) or self.n_char == '.':
                 if self.c_char == '.':
                     dots += 1
                 self.next()
             else:
                 break
         if dots > 1: 
             raise ValueError('Invalid number format')
         number_value = self.source[curr_pos: self.position + 1]
         cleaned = number_value.replace('-', '')
         if dots > 0:
             return Token(TokenKind.N_LITERAL, float(cleaned))
         return Token(TokenKind.N_LITERAL, int(cleaned))           
         
     def is_digit(self, char):
          return char in '0123456789'
     def is_letter(self, char):
         return char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
     def make_identifier(self):       
         curr_pos = self.position
         while True:
            if self.c_char != '\0' and self.c_char != ';' and not self.is_whitespace_char(self.c_char) and (self.n_char.isalnum() or self.n_char == '_'):
                #it could be TFL1 or TFL2...
                 self.next()
            else:
                if not self.is_whitespace_char(self.n_char) and self.is_digit(self.n_char):
                    #we have a TFL1 AG5 etc
                    self.next() #before we break... i.e eat that wandering number attached to an identifier too...
                 
                break
         literal = self.source[curr_pos:self.position+1]
         return Token(TokenKind.IDENTIFIER,literal)
     def make_string_literal(self):
         c_char = self.c_char # get that c_char whether it's ' or "
         self.next()
         curr_pos = self.position
         
         while True:
             if self.c_char != '\0' and self.c_char != c_char:
                 self.next()
             else:
                 break 
         literal = self.source[curr_pos:self.position - curr_pos]
         return Token(TokenKind.S_LITERAL, literal)
     def next(self):
         p_index = self.position 
         self.position += 1
         self.c_char = self.n_char
         if self.position <= (len(self.source) - 2):
             self.n_char = self.source[self.position + 1]
         else:
             self.n_char = '\0'
        
         
        
        


class StatementKind(Enum):
    VAR = 0
    FN = 1
    EMPTY = 2
             
class VarStatement:
    def __init__(self, name, expr):
        self.name = name
        self.expr = expr
        self.kind = StatementKind.VAR
        
    def __str__(self):
        return f'VAR {{ name: {self.name}, expr: {self.expr}'
    
class EmptyStatement:
    def __init__(self):
        self.name = '<EMPTY>'
        self.kind = StatementKind.EMPTY
    def __str__(self):
        return f'EMPTY {{ name: {self.name}'
class AstProgram:
    def __int__(self, statements):
        self.statements = statements
        
    def __str__(self):
        return f'[{", ".join(self.statements)}]'
    
    
class Parser:
    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.c_token = Token(TokenKind.EOF, '\0')
        self.n_token = Token(TokenKind.EOF, '\0')
        self.tokens = []
        
        self.next()
        
    def next(self):
        self.c_token = self.n_token
        self.n_token = self.lexer.lex()
        self.tokens.append(self.c_token)
    def parse(self):
        statements = []
        while True:
            c = self.c_token
            self.next()
            if self.c_token.kind == TokenKind.EOF:
                break
            stmt = self.parse_statement()
            if stmt is None:
                print(f'could not parse statement beginning at {self.lexer.position}')
                break
            statements.append(stmt)
        return statements
    
    def parse_statement(self):
        c = self.c_token
        if c.kind != TokenKind.EOF and c.kind != TokenKind.ENDLINE:
            # is_no_colon_var_decl_type = ':' not in self.lexer.source and self.last_nth_token(2).kind == TokenKind.ENDLINE or self.last_nth_token(2).kind == TokenKind.EOF
            is_no_colon_var_decl_type = ':' not in self.lexer.source
            if c.kind == TokenKind.IDENTIFIER and self.n_token.kind == TokenKind.ASSIGN and self.peek_next_n_token(2).kind == TokenKind.COLON or is_no_colon_var_decl_type:
                return self.parse_var_declaration(is_no_colon_var_decl_type)
            else:
                other_expr = self.parse_expr2(Precedence.LOWEST)
                return self.parse_expr_statement(other_expr)
        raise ValueError('invalid statement detected')
        return None
    def last_nth_token(self, n):
        '''
        This is a hack to let us know this is a starting statement i.e a VAR not just an assignment inside the RHS...
        This is to accomodate Dr Claus's kind of BN rules...that don't declare a new variable like this VAR =1 :
        Ben's own works perfectly already...
        '''
        # lex = copy.copy(self.lexer)
        # lex.positio
        # last_n_tok = lex.lex()  
        last_n_tok = self.tokens[len(self.tokens)-n]
        return last_n_tok
    def peek_next_n_token(self, n):
        toks = []
        lexer_copy = copy.copy(self.lexer)
        for i in range(n):
            toks.append(lexer_copy.lex())
        return toks[-1]
        
    def parse_expr_statement(self, other_expr):
        pass

    def parse_expr2(self, precedence):
        '''
        Parse any kind of expression with specified precedence
        '''
        left = ('', '')
        if self.c_token.kind == TokenKind.S_LITERAL:
            literal = self.c_token.literal
            self.expect_token_and_read(TokenKind.S_LITERAL)
            left = ("[STR_EXPR]", literal)
        elif self.c_token.kind == TokenKind.N_LITERAL:
            number = int(self.c_token.literal)
            self.expect_token_and_read(TokenKind.N_LITERAL)
            left = ('[NUM_EXPR]', number)
        elif self.c_token.kind == TokenKind.IDENTIFIER and self.n_token.kind == TokenKind.ASSIGN:
            id_token = self.expect_identifier_and_read()
            self.expect_token_and_read(TokenKind.ASSIGN)
            number = self.expect_token_and_read(TokenKind.N_LITERAL)
            left = ('[ASSIGNMENT_EXPR]', id_token.literal, int(number.literal))
        elif self.c_token.kind == TokenKind.IDENTIFIER:
            id_token = self.expect_identifier_and_read()
            left = ('[IDENT_EXPR]', id_token.literal)
        elif self.c_token.kind == TokenKind.ASSIGN:
            literal = self.c_token.literal
            left = ('[ASSIGN_EXPR]', literal)
        elif self.c_token.kind in [TokenKind.AMPERSAND, TokenKind.AND, TokenKind.LOGICAL_AND]:
            literal = self.c_token.literal
            self.expect_token_and_read(self.c_token.kind)
            left = ('[PREFIX_AND]', literal, self.parse_expr2(Precedence.PREFIX))
        elif self.c_token.kind in [TokenKind.NOT, TokenKind.NOT_EQUAL, TokenKind.LOGICAL_NOT]:
            literal = self.c_token.literal
            self.expect_token_and_read(self.c_token.kind)
            left = ('[PREFIX_NOT]', literal, self.parse_expr2(Precedence.PREFIX)) #shoudl we separate not from notEq
        elif self.c_token.kind == TokenKind.L_PAREN:
            stack = []
            self.expect_token_and_read(TokenKind.L_PAREN)
            stack.append(TokenKind.L_PAREN)


            inner_expr = self.parse_expr2(Precedence.LOWEST)
            self.expect_token_and_read(TokenKind.R_PAREN)
            left = ('[PAREN_EXPR]', inner_expr)
               
        while True:
            if self.c_token.kind != TokenKind.EOF and self.c_token.kind != TokenKind.ENDLINE and precedence.value <= self.convert_kind_to_precedence(self.c_token.kind).value:                    
                postfix_expr = self.parse_postfix_expr(left)
                infix_expr = self.parse_infix_expr(left)
                
                if postfix_expr is not None:
                    left = postfix_expr
                elif infix_expr is not None:
                    left = infix_expr
                else:
                    break
            else:
                break            
        return left
    
    def parse_postfix_expr(self, left):
        #should we append an invisible L_PAREN...            
        if self.c_token.kind == TokenKind.L_PAREN:
            self.expect_token_and_read(TokenKind.L_PAREN)
            args = [] #list of expr
            while self.c_token.kind != TokenKind.R_PAREN:
                args.append(self.parse_expr2(Precedence.LOWEST))
                if self.c_token.kind == TokenKind.COMMA:
                    self.next()
                    
            self.expect_token_and_read(TokenKind.R_PAREN)
            return ('[CALL_EXPR]', left, args) #expr_name, left expr, arguments 
        else:
            return None
   
    def parse_infix_expr(self, left):
        if self.c_token.kind in [TokenKind.PLUS, TokenKind.MINUS, TokenKind.STAR, 
                                 TokenKind.EQUALITY, TokenKind.NOT_EQUAL, TokenKind.LOGICAL_NOT, TokenKind.LESS_OR_EQUAL, TokenKind.LESS_THAN, TokenKind.GREATER_THAN, TokenKind.GREATER_OR_EQUAL, TokenKind.AND, TokenKind.AMPERSAND, TokenKind.LOGICAL_AND, TokenKind.LOGICAL_OR, TokenKind.OR]:
            token = self.c_token
            self.next()
            right = self.parse_expr2(self.convert_kind_to_precedence(token.kind))
            return ('[INFIX_EXPR]', left, self.convert_kind_to_operand(token.kind), right)
        elif self.c_token.kind == TokenKind.ASSIGN:
            self.next()
            right = self.parse_expr2(Precedence.ASSIGN)
            return ('[ASSIGN_EXPR]', left, right)
        else:
            return None
                            
    def convert_kind_to_operand(self, token_kind):
        if token_kind == TokenKind.PLUS:
            return 'ADD'
        elif token_kind == TokenKind.MINUS:
            return 'SUBTRACT'
        elif token_kind == TokenKind.STAR:
            return 'MULTIPLY'
        elif token_kind == TokenKind.ASSIGN:
            return 'ASSIGN'
        elif token_kind == TokenKind.LESS_THAN:
            return "LESS_THAN"
        elif token_kind == TokenKind.GREATER_THAN:
            return 'GREATER_THAN'
        elif token_kind == TokenKind.LESS_OR_EQUAL:
            return 'LESS_OR_EQUAL'
        elif token_kind == TokenKind.GREATER_OR_EQUAL:
            return 'GREATER_OR_EQUAL'
        elif token_kind in [TokenKind.AND, TokenKind.LOGICAL_AND, TokenKind.AMPERSAND]:
            return 'AND'
        elif token_kind in [TokenKind.OR, TokenKind.LOGICAL_OR]:
            return 'OR'
        elif token_kind in [TokenKind.NOT, TokenKind.NOT_EQUAL, TokenKind.LOGICAL_NOT]:
            return 'NOT_EQUAL'
        elif token_kind == TokenKind.EQUALITY:
            return 'EQUALS'
        
        raise ValueError(f'Invalid token kind: {token_kind}')
        
    def convert_kind_to_precedence(self, token_kind):
        if token_kind in [TokenKind.STAR]:
            return Precedence.PRODUCT
        elif token_kind in [TokenKind.PLUS, TokenKind.MINUS]:
            return Precedence.SUM
        elif token_kind in [TokenKind.L_PAREN, TokenKind.R_PAREN, TokenKind.DOT, TokenKind.L_SQUARE]:
            return Precedence.CALL
        elif token_kind in [TokenKind.LESS_THAN, TokenKind.GREATER_THAN, TokenKind.LESS_OR_EQUAL, TokenKind.GREATER_OR_EQUAL]:
            return Precedence.LESS_THAN_OR_GREATER_THAN
        elif token_kind in [TokenKind.EQUALITY, TokenKind.NOT_EQUAL, TokenKind.NOT, TokenKind.LOGICAL_NOT]:
            return Precedence.EQUALS
        elif token_kind in [TokenKind.AND, TokenKind.LOGICAL_AND, TokenKind.LOGICAL_OR, TokenKind.OR]:
            return Precedence.AND_OR
        elif token_kind == TokenKind.ASSIGN:
            return Precedence.ASSIGN
        else:
            return Precedence.LOWEST
            
    def parse_var_declaration(self, is_no_colon_var_decl_type=False):
        '''
        Parse a variable declaration
        '''
        # self.expect_token_and_read(TokenKind.IDENTIFIER)
        name = self.expect_identifier_and_read().literal    
        if self.c_token.kind == TokenKind.ASSIGN:
            _ = self.expect_token_and_read(TokenKind.ASSIGN)
            value = '1' if is_no_colon_var_decl_type else self.expect_token_and_read(TokenKind.N_LITERAL).literal #eat the number OUTPUT value for this socalled identifier variable...
            if not is_no_colon_var_decl_type: self.expect_token_and_read(TokenKind.COLON)
            initial = self.parse_expr2(Precedence.LOWEST)
            return ('[VAR_EXPR]', name, int(value), initial)
        
        else:
            return EmptyStatement()
            
    def expect_token_and_read(self, token_kind):
        result = self.expect_token_and_throws(token_kind)
        self.next()
        return result
    def expect_identifier_and_read(self):
        return self.expect_token_and_read(TokenKind.IDENTIFIER)
    def expect_identifier(self):
        return self.expect_token(TokenKind.IDENTIFIER)
    def expect_token(self, token_kind):
        result = self.expect_token_and_throws(token_kind)
        self.next()
        return result
    def expect_token_and_throws(self, token_kind):
        if self.c_token.kind != token_kind:
            raise ValueError(f'Unexpected token exception {self.c_token.literal}')
        return self.c_token
    
class Evaluator:
    def __init__(self):
        self.env = {}  
        self.constants = {}
        self.store = [] #list of string of the python code I generate from the multi-state rules defined...       
        
    def compile(self, ast):
        #aggregates compilation unit gotten from compiling these statements in the 
        #passed program which is basically statements of these BN or multi-state rules...
        for node in ast:
            self.evaluate(node)  
            
        #here we've collected all the compiled symbols now we want to cleanup if if ifs, 
        #and if's or ifs and all other garbage generated cos my compiler is not so perfect yet...
        symbols = self.cleanup_symbols()
            
        # extract constants...out of there
        self.constants = self.extract_constants(symbols)
            
        
        return self.env, self.constants, symbols
    
    def cleanup_symbols(self):
        import re
        cleaned_list = []
        
        for code in self.store:
            code = re.sub(r'\b(if\s+)+', 'if ', code)
            
            code = re.sub(r'\b(or|and)\s+if\b', r'\1', code)
            
            if re.search(r'^\s*not\s+\w+\s*:', code):  
                code = 'if ' + code
            elif re.search(r'^\s*\w+\s*:', code):
                code = 'if ' + code
            
            cleaned_list.append(code.strip())
    
        return cleaned_list

    def extract_constants(self, expressions):
        import re
        from collections import defaultdict
        constants = defaultdict(set)
        
        for expression in expressions:
            # Extract assignments from the right-hand side of the ':'
            if ':' in expression:
                conditions, assignments = expression.split(':', 1)
            else:
                conditions, assignments = expression, ""
            
            # Find all variable assignments in the right-hand side
            for assign in assignments.split('or'):
                assign = assign.strip()
                if '==' in assign or '=' not in assign:
                    continue
                var, val = assign.split('=')
                constants[var.strip()].add(int(val.strip()))
            
            # Extract variables and their values from the left-hand side
            for match in re.findall(r'([A-Za-z0-9_]+) *== *([0-9]+)', conditions):
                var, val = match
                constants[var].add(int(val))
        
        return {key: sorted(values) for key, values in constants.items()}
    def evaluate(self, node):
        """Recursively evaluate an AST node."""
        node_type = node[0]

        if node_type == "[VAR_EXPR]":
            return self.evaluate_var_expr(node)
        elif node_type == "[ASSIGNMENT_EXPR]":
            return self.evaluate_assignment_expr(node)
        elif node_type == "[INFIX_EXPR]":
            return self.evaluate_infix_expr(node)
        elif node_type == "[PAREN_EXPR]":
            return self.evaluate_paren_expr(node)
        elif node_type == "[IDENT_EXPR]":
            return self.evaluate_ident_expr(node)
        elif node_type == "[PREFIX_NOT]":
            
            return self.evaluate_prefix_expr(node)
        elif node_type == "[NUM_EXPR]":
            return node[1]
        else:
            raise ValueError(f"Unknown node type: {node_type}")
            
    def evaluate_prefix_expr(self, node):
        """Evaluate an identifier: [IDENT_EXPR, var_name]"""
        _, _, right = node
        right_val = self.evaluate(right)
        if right_val == 'IkB':
           print('We are here...')
        return f'not {right_val}'
        
    def evaluate_var_expr(self, node):
        """Evaluate a variable definition rule: [VAR_EXPR, name, value, condition]"""
        _, name, value, condition = node
        
        if name in self.env:
            self.env[name].append(value)
        else:
            self.env[name] = [value]
        
        exp_type = condition[0]
        if exp_type != '[ASSIGNMENT_EXPR]':
            condition_result = self.evaluate(condition)
            self.store .append(f'{condition_result}: {name}={value}')
        else:
            condition_result = self.evaluate(condition)
            self.store.append(f'if {condition_result}: {name}={value}')
        return condition_result

    def evaluate_assignment_expr(self, node, is_normal_assignment=False):
        """Evaluate an assignment: [ASSIGNMENT_EXPR, var_name, value]"""
        _, var_name, value = node      
        return f'{var_name} = {value}' if is_normal_assignment else f'{var_name} == {value}'

    def evaluate_infix_expr(self, node):
        """Evaluate infix operations like AND/OR: [INFIX_EXPR, left, op, right]"""
        _, left, op, right = node
        left_val = self.evaluate(left)
        right_val = self.evaluate(right)
        if op == "AND":
            return f'if {left_val} and {right_val}'
        elif op == "OR":
            return f'if {left_val} or {right_val}'
        else:
            raise ValueError(f"Unsupported operator: {op}")

    def evaluate_paren_expr(self, node):
        """Evaluate parentheses: [PAREN_EXPR, inner_expr]"""
        _, inner_expr = node
        return self.evaluate(inner_expr)

    def evaluate_ident_expr(self, node):
        """Evaluate an identifier: [IDENT_EXPR, var_name]"""
        _, var_name = node
        return f'{var_name}'
        # return self.env.get(var_name, 0)  # Default to 0 if not found

 
if __name__=='__main__':
    # source = """
    # TrpE=1 :	TrpR=0 AND Trp=0
    # TrpR=1 :	Trp=2
    # Trp=1 :	(Trpext=0 AND TrpE=1) OR (Trpext=1 AND TrpE=0) OR (Trpext=1 AND TrpE=1)
    # Trp=2 :	(Trpext=2 AND TrpE=0) OR (Trpext=2 AND TrpE=1)
    # """
    # source = '''TrpE=1 :	TrpR=0 AND Trp=0'''
    # source = '''TrpR=1 :	Trp=2
    # TrpE=1 :	TrpR=0 AND Trp=0
    # TrpE=1 :	(TrpR=0 AND Trp=0)
    # '''
    
    # '16204102.txt' '15486106.txt'
    # source = 'HGF_c_Met = (HGF AND c_Met) AND NOT LY3'  
    source = '''HGF_c_Met = (HGF AND c_Met)
    Gab1 = (HGF_c_Met)
    SHP2 = (Gab1) OR (HGF_c_Met) OR (CagA)
    Grb2 = (HGF_c_Met) OR (Shc1)
    PI3K = (Gab1) OR (c_Cbl)
    c_Cbl = (HGF_c_Met)
    Crk_CrkL = (Gab1) OR (c_Cbl) OR (CagA)
    Shc1 = (HGF_c_Met)
    DOCK180 = (Crk_CrkL)
    Rac1 = (DOCK180)
    STAT3 = (RAC1) OR (HGF_c_Met)
    ILK = (PI3K)
    PIP3 = (NOT PTEN AND PI3K AND PIP2)
    PDK1 = (PIP3)
    AKT1 = (PDK1 AND ILK)
    RAC1 = (AKT1)
    PAK1 = (AKT1) OR (Rac1)
    IQGAP_1 = (Rac1 AND Calmodulin)
    MKK4 = (PAK1)
    JNK = (MKK4)
    c_JUN = (JNK)
    ELK1 = (JNK) OR (ERK1_2)
    ATF2 = (JNK)
    IKK = (PAK1)
    IκBalpha = (NOT IKK)
    NF_κB = (NOT IκBalpha)
    RasGAP = (NOT SHP2 AND Gab1)
    SOS1 = (Grb2)
    Ras = (NOT RasGAP AND SOS1)
    Raf1 = (Ras) OR (PKCalpha)
    MEK = (Raf1)
    ERK1_2 = (MEK)
    c_Myc = (ERK1_2)
    ETS = (ERK1_2)
    PLCgamma1 = (Gab1) OR (CagA_c_Met)
    IP3 = (PLCgamma1 AND PIP2)
    DAG = (PLCgamma1 AND PIP2)
    Ca2p = (IP3)
    PKCalpha = (DAG AND Ca2p)
    Calmodulin = (Ca2p)
    alpha_catenin_beta_catenin_E_cadherin = (NOT IQGAP_1)
    CagA = (c_SRC AND FYN AND H_pylori)
    CSK = (NOT H_pylori) OR (CagA)
    c_SRC = (NOT CSK)
    FYN = (NOT CSK)
    FAK = (NOT SHP2)
    PIPKIgamma661 = (FAK)
    Talin = (PIPKIgamma661)
    Calpain2 = (FAK AND ERK1_2)
    CagA_c_Met = (CagA AND c_Met)
'''      
    # source = '''
    # T2 = TNF AND NOT FLIP
    # IKKa = TNF AND NOT A20a AND NOT C3a
    # NFkB = NOT IkB
    # NFkBnuc = NFkB AND NOT IkB
    # IkB = (TNF AND NFkBnuc AND NOT IKKa) OR (NOT TNF AND (NFkBnuc OR NOT IKKa))
    # A20a = TNF AND NFkBnuc
    # IAP = (TNF AND NFkBnuc AND NOT C3a) OR (NOT TNF AND (NFkBnuc OR NOT C3a))
    # FLIP = NFkBnuc
    # C3a = NOT IAP AND C8a
    # C8a = NOT CARP AND (C3a OR T2)
    # CARP = (TNF AND NFkBnuc AND NOT C3a) OR (NOT TNF AND (NFkBnuc OR NOT C3a))
    # '''
    source = '' #19524598.txt 
    with open('16204102.txt', 'r', encoding='utf-8') as f:
        source += f.read()

    # print(f'New Source after stripping parentheses >>> {source}')
    # source = 'AG=0 :	((((( (( TFL1=1 AND LFY=0 ) OR ( TFL1=2 AND LFY=0 ) OR ( TFL1=2 AND LFY=1 )) ) OR ( (( TFL1=0 AND LFY=0 ) OR ( TFL1=1 AND LFY=1 )) AND AP2=1 )) OR ( (( LFY=1 AND TFL1=0 ) OR ( LFY=2 AND TFL1=0 ) OR ( LFY=2 AND TFL1=1 )) AND ( AP1=1 OR AP1=2 ) AND ( AG=0 OR AG=1 ) AND WUS=0 AND AP2=1 AND LUG=1 AND CLF=1  )) OR ( (( LFY=1 AND TFL1=0 ) OR ( LFY=2 AND TFL1=0 ) OR ( LFY=2 AND TFL1=1 )) AND ( AP1=1 OR AP1=2 ) AND AG=2 AND WUS=0 AND AP2=1 AND SEP=0 AND LUG=1 AND CLF=1  )) OR ( TFL1=2 AND LFY=2 AND ( AP1=1 OR AP1=2 ) AND ( AG=0 OR AG=1 ) AND WUS=0 AND AP2=1 AND LUG=1 AND CLF=1 )) OR ( TFL1=2 AND LFY=2 AND ( AP1=1 OR AP1=2 ) AND AG=2 AND WUS=0 AND AP2=1 AND SEP=0 AND LUG=1 AND CLF=1 )'
    # source = 'HGF_c_Met = (HGF AND c_Met) AND NOT LY3' 
    # source = 'Crk_CrkL = (Gab1) OR (c_Cbl) OR (CagA)'
    tokens = [()]
    lexer = Lexer(source)
    parser = Parser(lexer)
    ast = parser.parse()
    print(f'Input model >>> {source}')
    print(f'The AST is now >>> {ast}')
    evaluator = Evaluator()
    variables, constants, symbols = evaluator.compile(ast)
    print(f'\n\nvariables: {variables}\n\n')
    print(f'\n\constants: {constants}\n\n')
    print(f'\n\n{symbols}\n')
    
 