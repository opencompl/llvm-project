irdl.dialect @test_irdl_to_cpp {
    irdl.type @foo {
        %0 = irdl.any
        irdl.parameters(type_param: %0)
    }
    
    irdl.operation @bar {
        %0 = irdl.any
        irdl.operands(some_operand: %0, another_operand: %0)
        irdl.results(a_result: %0)
    }
}
