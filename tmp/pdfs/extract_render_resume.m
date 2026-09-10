#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>
#import <PDFKit/PDFKit.h>

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 3) {
            fprintf(stderr, "usage: %s input.pdf output_dir\n", argv[0]);
            return 2;
        }

        NSString *inputPath = [NSString stringWithUTF8String:argv[1]];
        NSString *outputDir = [NSString stringWithUTF8String:argv[2]];
        PDFDocument *document = [[PDFDocument alloc]
            initWithURL:[NSURL fileURLWithPath:inputPath]];
        if (document == nil) {
            fprintf(stderr, "failed to open PDF\n");
            return 1;
        }

        printf("PAGES: %ld\n", (long)document.pageCount);
        for (NSInteger index = 0; index < document.pageCount; ++index) {
            PDFPage *page = [document pageAtIndex:index];
            NSRect bounds = [page boundsForBox:kPDFDisplayBoxMediaBox];
            printf("\n===== PAGE %ld (%.0f x %.0f pt) =====\n",
                   (long)index + 1, bounds.size.width, bounds.size.height);
            NSString *text = page.string ?: @"";
            printf("%s\n", text.UTF8String);

            NSSize imageSize = NSMakeSize(bounds.size.width * 2.0,
                                          bounds.size.height * 2.0);
            NSImage *image = [page thumbnailOfSize:imageSize
                                             forBox:kPDFDisplayBoxMediaBox];
            NSBitmapImageRep *bitmap = [NSBitmapImageRep
                imageRepWithData:image.TIFFRepresentation];
            NSData *png = [bitmap representationUsingType:NSBitmapImageFileTypePNG
                                                properties:@{}];
            NSString *filename = [NSString stringWithFormat:@"page-%ld.png",
                                                              (long)index + 1];
            NSString *outputPath = [outputDir stringByAppendingPathComponent:filename];
            if (![png writeToFile:outputPath atomically:YES]) {
                fprintf(stderr, "failed to write %s\n", outputPath.UTF8String);
                return 1;
            }
        }
    }
    return 0;
}
