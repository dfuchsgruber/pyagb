OUTPUT_ARCH(arm)
MEMORY {

        rom     : ORIGIN = LOCATION, LENGTH = 32M
        ewram   : ORIGIN = 0x02000000, LENGTH = 4M - 4k
}

SECTIONS {
        .text : {

                FILL (0xABCD)
                __text_start = . ;
                *(.init)
				*(.text.startup)
                *(.text)
                *(.ctors)
                *(.dtors)
                *(.rodata)
                *(.fini)
                *(COMMON)
                __text_end  = . ;
                *(.data)
                __bss_start__ = . ;
                *(.bss)
                __bss_end__ = . ;
                __file_end__ = . ;
        _end = __file_end__ ;
        __end__ = __file_end__ ;
        } >rom = 0xff
}
