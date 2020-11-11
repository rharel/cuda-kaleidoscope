#include <iomanip>
#include <iostream>
#include <time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>

#include "config.h"
#include "cudu.h"
#include "raytracer.h"

const char MAIN_WINDOW_TITLE[] = "Kaleidoscope";

cudu::host::Array3D<unsigned char> generated_image()
{
    std::cout << "Generating image..." << std::endl;

    cudu::host::Array3D<unsigned char> image = kaleido::raytraced_image(kaleido::Config());

    const cv::Mat image_cv(
        int(image.shape()[0]),
        int(image.shape()[1]),
        CV_8UC3,
        image.begin()
    );
    cv::namedWindow(MAIN_WINDOW_TITLE, cv::WINDOW_KEEPRATIO);
    cv::imshow(MAIN_WINDOW_TITLE, image_cv);
    cv::waitKey(30);

    std::cout << "Ready" << std::endl;

    return image;
}

void save_image(const cudu::host::Array3D<unsigned char>& image)
{
    const time_t time = std::time(nullptr);
    tm local_time;
    localtime_s(&local_time, &time);

    std::ostringstream oss;
    oss << std::put_time(&local_time, "%Y_%m_%d_%H_%M_%S");
    const std::string time_str = oss.str();

    const std::string file_name = "world_" + time_str + ".png";

    const cv::Mat image_cv(
        int(image.shape()[0]),
        int(image.shape()[1]),
        CV_8UC3,
        const_cast<unsigned char*>(image.begin())
    );
    cv::imwrite(file_name, image_cv);

    std::cout << "Saved " + file_name << std::endl;
}

void run_gui_main_loop(const cudu::host::Array3D<unsigned char>& image)
{
    constexpr char KEYCODE_ESCAPE = 27;
    constexpr char KEYCODE_SAVE = 's';
    char keycode = 0;

    while (cv::getWindowProperty(MAIN_WINDOW_TITLE, 0) >= 0 &&
           keycode != KEYCODE_ESCAPE)
    {
        switch (keycode = cv::waitKey(30))
        {
            case KEYCODE_SAVE: save_image(image); break;
            default: break;
        }
    }
}

int main(int argc, char* argv[])
{
    const cudu::host::Array3D<unsigned char> image = generated_image();
    run_gui_main_loop(image);

    return EXIT_SUCCESS;
}
