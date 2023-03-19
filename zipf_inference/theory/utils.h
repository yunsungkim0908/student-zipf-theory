#include <stdio.h>
#include <iostream>

#include <indicators/block_progress_bar.hpp>
#include <indicators/cursor_control.hpp>

#include <sys/stat.h>

#ifndef UTILS_H
#define UTILS_H

template <typename T>
void printVec(T* theArray, int N) {
    for ( int x = 0; x < N; x ++ ) {
        std::cout << theArray[x] << " ";
    }
    std::cout << std::endl;
}

template <typename T>
void printArray(T** theArray, int N, int M ) {
    for ( int x = 0; x < N; x ++ ) {
        for (int y = 0; y < M; y++) {
            std::cout << theArray[x][y] << " ";
        }
        std::cout << std::endl;
    }
}

inline indicators::BlockProgressBar* getProgressBar(){
    indicators::BlockProgressBar* bar = new indicators::BlockProgressBar {
        indicators::option::BarWidth{50},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true},
        indicators::option::FontStyles{
          std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}};
    return bar;
}

inline int dir_exists(const char* const path)
{
    /******************************************************************************
     * Checks to see if a directory exists. Note: This method only checks the
     * existence of the full path AND if path leaf is a dir.
     *
     * @return  >0 if dir exists AND is a dir,
     *           0 if dir does not exist OR exists but not a dir,
     *          <0 if an error occurred (errno is also set)
     *****************************************************************************/

    struct stat info;

    int statRC = stat( path, &info );
    if( statRC != 0 )
    {
        if (errno == ENOENT)  { return 0; } // something along the path does not exist
        if (errno == ENOTDIR) { return 0; } // something in path prefix is not a dir
        return -1;
    }

    return ( info.st_mode & S_IFDIR ) ? 1 : 0;
}

#endif
