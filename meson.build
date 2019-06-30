project('shisen', 'cpp',
    version : '0.1',
    default_options : [
        'cpp_std=c++17',
        'warning_level=2',
    ],
)

cpp_args = [
    '-Wconditionally-supported',
    '-Wconversion',
    '-Wduplicated-cond',
    '-Wduplicated-branches',
    '-Wextra-semi',
    '-Wfloat-equal',
    '-Wformat=2',
    '-Wlogical-op',
    '-Wnull-dereference',
    '-Wold-style-cast',
    '-Wshadow',
    '-Wswitch-default',
    '-Wswitch-enum',
    '-Wundef',
    '-Wuseless-cast',
    '-Wvla',
    '-Wzero-as-null-pointer-constant',
]
link_args = []

if not get_option('buildtype').startswith('release')
    cpp_args += [
        '-fsanitize=undefined',
        '-fno-sanitize-recover=all',
        '-D_GLIBCXX_DEBUG',
    ]
    link_args += [
        '-fsanitize=undefined',
        '-fno-sanitize-recover=all',
    ]
endif

dependencies = [
    dependency('fmt'),
    dependency('sdl2'),
    dependency('SDL2_image'),
]

executable('shisen',
    [
        'main.cpp',
    ],
    cpp_args : cpp_args,
    link_args : link_args,
    dependencies : dependencies,
)
