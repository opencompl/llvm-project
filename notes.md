notes:
- notifications are fired before the action is performed 
- in some patterns, dce is conducted after every pass, so there are op erasures without a corresponding rewrite pattern

issues with listener:
- you cannot detect a match using rewriter listener.
  - eg. muli %v, 2 lowered to addi, %v, %v does not remove the const 2.
  - however, we need the const op in the match. how should we detect this?

  solutions:
  - create an interface to emit a match hint
  - use PDL, because PDL declares the match, that uses the hint

questions:
- cursors
  - do we care about intermediate insertions?

- samply