#pragma once

namespace kaleido
{
    template <typename T>
    struct Size2D
    {
        T width = 0;
        T height = 0;
    };

    struct Config
    {
        float mirror_angle_degrees = 60;
        float mirror_albedo = 0.9;
        float background_texture_scale = 20;
        float camera_field_of_view_degrees = 120;
        float camera_background_distance = 30;
        size_t stratifier_resolution = 64;
        unsigned long stratifier_rng_seed = 0;
        size_t nr_render_iterations = 10000;
        Size2D<size_t> image_size = Size2D<size_t>{ 256, 256 };
    };
}
